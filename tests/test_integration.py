"""
Integration tests for the Raft election module (orchestrator package).

Each test spins up real ClusterNode instances communicating over real gRPC on
127.0.0.x loopback addresses (one IP per node, same port). This mirrors the
production topology - different IPs, same port - without requiring separate
machines.

Prerequisites:
  - server.py must bind on `host_ip`
  - Run with: uv run pytest packages/orchestrator/tests/test_integration.py -v
"""

import threading
import time
from typing import Optional

import pytest

from gradsync.orchestrator.node import ClusterNode, NodeState


# ---------------------------------------------------------------------------
# Test infrastructure — RaftCluster
# ---------------------------------------------------------------------------

# Use a port above 50051 to avoid conflicting with any running production node.
_TEST_PORT = 50090


class RaftCluster:
    """
    Test harness for a local multi-node Raft cluster.

    - Nodes communicate over real gRPC on 127.0.0.x loopback IPs.
    - Each node runs join_cluster() in its own daemon thread.
    - Helper methods mirror check_one_leader / check_terms etc.
    """

    @staticmethod
    def gen_local_instances(n: int, port: int = _TEST_PORT) -> list[ClusterNode]:
        """
        Generate N ClusterNode objects using 127.0.0.1 … 127.0.0.N on `port`.
        """
        ips = [f"127.0.0.{i + 1}" for i in range(n)]
        nodes = []
        for ip in ips:
            peers = [p for p in ips if p != ip]
            nodes.append(ClusterNode(host_ip=ip, peer_ips=peers, port=port))
        return nodes

    def __init__(self, nodes: list[ClusterNode]):
        self.nodes = nodes
        self._threads: list[threading.Thread] = []
        self._exceptions: dict[str, Exception] = {}

    def begin(self, stagger_delays: dict[str, float] = None):
        """
        Start all nodes concurrently. Each node runs join_cluster() in a
        daemon thread so the test process can exit even if a node hangs.
        If `stagger_delays` dictionary exists, delays execution by the mapped float seconds.
        """
        stagger_delays = stagger_delays or {}
        for node in self.nodes:
            delay = stagger_delays.get(node.host_ip, 0.0)
            t = threading.Thread(
                target=self._run_node,
                args=(node, delay),
                daemon=True,
                name=f"raft-{node.host_ip}",
            )
            self._threads.append(t)
            t.start()

    def _run_node(self, node: ClusterNode, delay: float):
        try:
            if delay > 0.0:
                time.sleep(delay)
            node.join_cluster()
        except Exception as e:
            self._exceptions[node.host_ip] = e

    def wait(self, timeout: float = 5.0):
        """
        Block until all nodes finish join_cluster(), up to `timeout` seconds.
        Fails the test if any node is still running or raised an exception.
        """
        try:
            for t in self._threads:
                t.join(timeout=timeout)
    
            if self._exceptions:
                pytest.fail(f"Nodes raised exceptions during join_cluster(): "
                            f"{self._exceptions}")

            still_running = [t.name for t in self._threads if t.is_alive()]
            if still_running:
                pytest.fail(
                    f"Nodes did not complete join_cluster() within {timeout}s: "
                    f"{still_running}"
                )
        finally:
            # Shutdown all nodes now that tests have completed their primary wait
            for node in self.nodes:
                node.shutdown()

    def check_one_leader(self) -> ClusterNode:
        """
        Assert exactly one node ended up as LEADER and return it.
        """
        leaders = [n for n in self.nodes if n.state == NodeState.LEADER]
        assert len(leaders) == 1, (
            f"Expected exactly 1 leader, got {len(leaders)}: "
            f"{[l.host_ip for l in leaders]}"
        )
        return leaders[0]

    def check_all_agree(self) -> str:
        """
        Assert all nodes agree on the same coordinator_ip and return it.
        """
        coordinators = {n.topology_config.coordinator_ip for n in self.nodes}
        assert len(coordinators) == 1, (
            f"Nodes disagree on coordinator_ip: {coordinators}"
        )
        return coordinators.pop()

    def check_terms(self) -> int:
        """
        Assert all nodes share the same current_term and return it.
        """
        terms = {n.current_term for n in self.nodes}
        assert len(terms) == 1, f"Nodes disagree on current_term: {terms}"
        return terms.pop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.timeout(8)
def test_single_node_elects_itself():
    """
    A cluster of size 1 requires no votes from peers. The single node must
    immediately elect itself as leader and set its own IP as coordinator.
    """
    nodes = RaftCluster.gen_local_instances(1)
    cluster = RaftCluster(nodes)
    cluster.begin()
    cluster.wait(timeout=3.0)

    leader = cluster.check_one_leader()
    assert leader.host_ip == "127.0.0.1:50090"
    assert nodes[0].coordinator_ip == "127.0.0.1:50090"
    ordered = list(nodes[0].topology_config.ordered_node_ips)
    assert ordered == ["127.0.0.1:50090"]


@pytest.mark.timeout(10)
def test_three_node_cluster_elects_one_leader():
    """
    A 3-node cluster must converge on exactly one LEADER.
    """
    nodes = RaftCluster.gen_local_instances(3)
    cluster = RaftCluster(nodes)
    cluster.begin()
    cluster.wait(timeout=5.0)

    cluster.check_one_leader()


@pytest.mark.timeout(10)
def test_all_nodes_agree_on_coordinator():
    """
    After election, every node's topology_config must name the same
    coordinator_ip, and that IP must belong to one of the cluster members.
    """
    nodes = RaftCluster.gen_local_instances(3)
    cluster = RaftCluster(nodes)
    cluster.begin()
    cluster.wait(timeout=5.0)

    coordinator = cluster.check_all_agree()
    all_ips = {n.host_ip for n in nodes}
    assert coordinator in all_ips, (
        f"Coordinator IP {coordinator!r} is not a known cluster member: {all_ips}"
    )


@pytest.mark.timeout(10)
def test_coordinator_ip_matches_leader():
    """
    The coordinator_ip agreed on by all nodes must be the same node that
    holds state == LEADER.
    """
    nodes = RaftCluster.gen_local_instances(3)
    cluster = RaftCluster(nodes)
    cluster.begin()
    cluster.wait(timeout=5.0)

    leader = cluster.check_one_leader()
    coordinator = cluster.check_all_agree()
    assert coordinator == leader.host_ip, (
        f"Coordinator IP ({coordinator}) does not match leader IP ({leader.host_ip})"
    )


@pytest.mark.timeout(10)
def test_topology_contains_all_node_ips():
    """
    Every node's topology must include all cluster members in ordered_node_ips,
    with no duplicates.
    """
    nodes = RaftCluster.gen_local_instances(3)
    cluster = RaftCluster(nodes)
    cluster.begin()
    cluster.wait(timeout=5.0)

    all_ips = {n.host_ip for n in nodes}
    for node in nodes:
        ordered = list(node.topology_config.ordered_node_ips)
        assert len(ordered) == 3, (
            f"Node {node.host_ip}: expected 3 IPs in topology, got {len(ordered)}"
        )
        assert set(ordered) == all_ips, (
            f"Node {node.host_ip}: topology IPs {set(ordered)} != cluster IPs {all_ips}"
        )


@pytest.mark.timeout(10)
def test_leader_is_first_in_topology():
    """
    The pipeline depends on ordered_node_ips[0] being the coordinator/HEAD.
    Every node's copy of the topology must have the leader's IP at index 0.
    """
    nodes = RaftCluster.gen_local_instances(3)
    cluster = RaftCluster(nodes)
    cluster.begin()
    cluster.wait(timeout=5.0)

    leader = cluster.check_one_leader()
    for node in nodes:
        ordered = list(node.topology_config.ordered_node_ips)
        assert ordered[0] == leader.host_ip, (
            f"Node {node.host_ip}: expected leader {leader.host_ip} at index 0, "
            f"got {ordered[0]}"
        )


@pytest.mark.timeout(10)
def test_all_nodes_agree_on_term():
    """
    After a clean election, all nodes must converge on the same current_term.
    """
    nodes = RaftCluster.gen_local_instances(3)
    cluster = RaftCluster(nodes)
    cluster.begin()
    cluster.wait(timeout=5.0)

    term = cluster.check_terms()
    assert term >= 1, f"Term should be at least 1 after an election, got {term}"


@pytest.mark.timeout(15)
def test_three_node_cluster_staggered_start():
    """
    Tests the pre-Raft sync barrier. Nodes start separated by delays (e.g. 1-2 seconds apart).
    The first node MUST block and wait for the last node to start instead of
    immediately timing out and failing to form a complete network.
    """
    nodes = RaftCluster.gen_local_instances(3)
    cluster = RaftCluster(nodes)
    
    # Stagger nodes by 0, 1, and 2 seconds respectively.
    delays = {
        "127.0.0.1:50090": 0.0,
        "127.0.0.2:50090": 5.0,
        "127.0.0.3:50090": 10.0
    }
    cluster.begin(stagger_delays=delays)
    
    # Needs a bigger timeout (5s normal + 2s max stagger delay + margin)
    cluster.wait(timeout=20.0)
    
    cluster.check_one_leader()
    cluster.check_all_agree()
    cluster.check_terms()


@pytest.mark.timeout(15)
def test_five_node_cluster_elects_one_leader():
    """
    Larger cluster (5 nodes) — needs 3 votes for majority.
    Tests that randomized timeouts resolve split votes across more candidates.
    """
    nodes = RaftCluster.gen_local_instances(5)
    cluster = RaftCluster(nodes)
    cluster.begin()
    cluster.wait(timeout=8.0)

    cluster.check_one_leader()
    cluster.check_all_agree()
    cluster.check_terms()
