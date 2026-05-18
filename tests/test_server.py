"""
Unit tests for ClusterServer (server.py).

These tests call RequestVote and BroadcastTopology directly as plain Python
methods — no gRPC, no network. A real ClusterNode is used as state storage so
that the server's mutations are exercised against the actual production object.
"""

import threading
import time

import pytest

from gradsync.orchestrator.node import ClusterNode, NodeState
from gradsync.orchestrator.server import ClusterServer
from gradsync.orchestrator.proto import cluster_service_pb2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vote_request(term: int, candidate_ip: str) -> cluster_service_pb2.VoteRequest:
    return cluster_service_pb2.VoteRequest(term=term, candidate_ip=candidate_ip)


def make_topology(
    coordinator_ip: str, ordered_ips: list[str], target_ip: str = "10.0.0.1:50051"
) -> cluster_service_pb2.TopologyConfig:
    idx = ordered_ips.index(target_ip)
    prev_idx = idx - 1
    next_idx = idx + 1
        
    return cluster_service_pb2.TopologyConfig(
        coordinator_ip=coordinator_ip,
        ordered_node_ips=ordered_ips,
        node_index=idx,
        prev_node_idx=prev_idx,
        next_node_idx=next_idx
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node() -> ClusterNode:
    """Fresh FOLLOWER node, term=0, no vote cast, two peers."""
    return ClusterNode(host_ip="10.0.0.1:50051", peer_ips=["10.0.0.2:50051", "10.0.0.3:50051"])


@pytest.fixture
def server(node: ClusterNode) -> ClusterServer:
    return ClusterServer(node)


# ---------------------------------------------------------------------------
# RequestVote tests
# ---------------------------------------------------------------------------

class TestRequestVote:
    def test_grants_vote_on_first_request(self, server, node):
        """A follower with no prior vote should grant the first valid request."""
        req = make_vote_request(term=1, candidate_ip="10.0.0.2:50051")
        resp = server.RequestVote(req, context=None)

        assert resp.vote_granted is True
        assert resp.term == 1
        assert node.voted_for == "10.0.0.2:50051"
        assert node.current_term == 1

    def test_rejects_stale_term(self, server, node):
        """A candidate with an older term must be rejected outright."""
        node.current_term = 5

        req = make_vote_request(term=3, candidate_ip="10.0.0.2:50051")
        resp = server.RequestVote(req, context=None)

        assert resp.vote_granted is False
        assert node.current_term == 5          # Term must NOT be downgraded
        assert node.voted_for is None          # No vote cast

    def test_steps_down_and_grants_vote_on_higher_term(self, server, node):
        """
        Raft rule: if a RequestVote arrives with a term > ours, we must
        immediately revert to FOLLOWER, update our term, clear voted_for,
        and then evaluate the vote grant.
        """
        node.current_term = 2
        node.state = NodeState.CANDIDATE
        node.voted_for = "10.0.0.1:50051"  # voted for self in a previous round

        req = make_vote_request(term=5, candidate_ip="10.0.0.2:50051")
        resp = server.RequestVote(req, context=None)

        assert node.state == NodeState.FOLLOWER
        assert node.current_term == 5
        assert node.voted_for == "10.0.0.2:50051"
        assert resp.vote_granted is True

    def test_vote_is_idempotent_for_same_candidate(self, server, node):
        """Voting for the same candidate twice in the same term is allowed."""
        req = make_vote_request(term=1, candidate_ip="10.0.0.2:50051")

        resp1 = server.RequestVote(req, context=None)
        resp2 = server.RequestVote(req, context=None)

        assert resp1.vote_granted is True
        assert resp2.vote_granted is True
        assert node.voted_for == "10.0.0.2:50051"

    def test_rejects_second_candidate_in_same_term(self, server, node):
        """Once a vote is cast for candidate A in term T, candidate B is rejected."""
        req_a = make_vote_request(term=1, candidate_ip="10.0.0.2:50051")
        req_b = make_vote_request(term=1, candidate_ip="10.0.0.3:50051")

        resp_a = server.RequestVote(req_a, context=None)
        resp_b = server.RequestVote(req_b, context=None)

        assert resp_a.vote_granted is True
        assert resp_b.vote_granted is False
        assert node.voted_for == "10.0.0.2:50051"   # Must not switch to B

    def test_returns_current_term_in_response(self, server, node):
        """The response term must always reflect the node's up-to-date term."""
        node.current_term = 7
        req = make_vote_request(term=7, candidate_ip="10.0.0.2:50051")
        resp = server.RequestVote(req, context=None)

        assert resp.term == 7


# ---------------------------------------------------------------------------
# BroadcastTopology tests
# ---------------------------------------------------------------------------

class TestBroadcastTopology:
    def test_sets_topology_config(self, server, node):
        """Receiving a topology should persist it on the node."""
        topo = make_topology("10.0.0.2:50051", ["10.0.0.2:50051", "10.0.0.1:50051", "10.0.0.3:50051"])
        resp = server.BroadcastTopology(topo, context=None)

        assert resp.ok is True
        assert node.topology_config is not None
        assert node.topology_config.coordinator_ip == "10.0.0.2:50051"
        # Assert the indices are parsed and passed flawlessly through into memory
        assert node.topology_config.node_index == 1
        assert node.topology_config.prev_node_idx == 0
        assert node.topology_config.next_node_idx == 2

    def test_sets_coordinator_ip(self, server, node):
        topo = make_topology("10.0.0.2:50051", ["10.0.0.2:50051", "10.0.0.1:50051", "10.0.0.3:50051"])
        server.BroadcastTopology(topo, context=None)

        assert node.coordinator_ip == "10.0.0.2:50051"

    def test_reverts_state_to_follower(self, server, node):
        """A candidate receiving a topology must accept it and step down."""
        node.state = NodeState.CANDIDATE

        topo = make_topology("10.0.0.2:50051", ["10.0.0.2:50051", "10.0.0.1:50051"])
        server.BroadcastTopology(topo, context=None)

        assert node.state == NodeState.FOLLOWER

    def test_notifies_waiting_thread(self, server, node):
        """
        join_cluster() blocks on _election_cv.wait(). BroadcastTopology must
        call notify_all() so that waiting thread can exit its loop.
        """
        woken = threading.Event()

        def waiter():
            with node._election_cv:
                node._election_cv.wait(timeout=5.0)
            woken.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.05)  # Ensure the thread reaches wait() before we notify

        topo = make_topology("10.0.0.2:50051", ["10.0.0.2:50051", "10.0.0.1:50051"])
        server.BroadcastTopology(topo, context=None)

        t.join(timeout=2.0)
        assert woken.is_set(), "BroadcastTopology did not wake the election thread"

    def test_ack_response_is_true(self, server, node):
        """The Ack.ok field must be True on success."""
        topo = make_topology("10.0.0.3:50051", ["10.0.0.3:50051", "10.0.0.1:50051", "10.0.0.2:50051"])
        resp = server.BroadcastTopology(topo, context=None)

        assert resp.ok is True
