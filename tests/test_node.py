"""
Unit tests for ClusterNode (node.py).

ClusterClient is mocked throughout — no real gRPC channels are opened.
This isolates the Raft state machine logic from network I/O entirely.
"""

import pytest
from unittest.mock import patch, MagicMock, call

from gradsync.orchestrator.node import ClusterNode, NodeState
from gradsync.orchestrator.proto import cluster_service_pb2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_node() -> ClusterNode:
    """Node as part of a 3-node cluster."""
    return ClusterNode(host_ip="10.0.0.1:50051", peer_ips=["10.0.0.2:50051", "10.0.0.3:50051"])

# ---------------------------------------------------------------------------
# IP Normalization
# ---------------------------------------------------------------------------

class TestIPNormalization:
    def test_appends_default_port_to_raw_ips(self):
        node = ClusterNode(host_ip="10.0.0.1", peer_ips=["10.0.0.2", "10.0.0.3"])
        assert node.host_ip == "10.0.0.1:50051"
        assert node.peer_ips == ["10.0.0.2:50051", "10.0.0.3:50051"]

    def test_preserves_explicit_ports(self):
        node = ClusterNode(host_ip="10.0.0.1:8080", peer_ips=["10.0.0.1:8081", "10.0.0.2"])
        assert node.host_ip == "10.0.0.1:8080"
        assert node.peer_ips == ["10.0.0.1:8081", "10.0.0.2:50051"]


# ---------------------------------------------------------------------------
# Server Binding
# ---------------------------------------------------------------------------

class TestServerBinding:
    @patch("orchestrator.node.grpc.server")
    def test_raises_runtime_error_on_port_collision(self, mock_grpc_server, three_node):
        mock_server_instance = MagicMock()
        mock_grpc_server.return_value = mock_server_instance
        mock_server_instance.add_insecure_port.return_value = 0

        with pytest.raises(RuntimeError, match="Failed to bind to 10.0.0.1:50051"):
            three_node._serve_cluster()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_as_follower(self, three_node):
        assert three_node.state == NodeState.FOLLOWER

    def test_starts_at_term_zero(self, three_node):
        assert three_node.current_term == 0

    def test_no_vote_cast(self, three_node):
        assert three_node.voted_for is None

    def test_no_coordinator(self, three_node):
        assert three_node.coordinator_ip is None

    def test_no_topology(self, three_node):
        assert three_node.topology_config is None

    def test_zero_votes_received(self, three_node):
        assert three_node.votes_received == 0


# ---------------------------------------------------------------------------
# send_request_vote
# ---------------------------------------------------------------------------

class TestSendRequestVote:
    def test_returns_true_when_vote_granted(self, three_node):
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.request_vote.return_value = (True, 1)
            result = three_node.send_request_vote("10.0.0.2:50051", term=1, candidate_ip="10.0.0.1:50051")

        assert result is True

    def test_returns_false_when_vote_denied(self, three_node):
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.request_vote.return_value = (False, 1)
            result = three_node.send_request_vote("10.0.0.2:50051", term=1, candidate_ip="10.0.0.1:50051")

        assert result is False

    def test_steps_down_when_peer_has_higher_term(self, three_node):
        """
        Raft rule: if any peer responds with term > ours during an election,
        we must immediately revert to FOLLOWER and clear voted_for.
        """
        three_node.current_term = 1
        three_node.state = NodeState.CANDIDATE

        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.request_vote.return_value = (False, 10)
            three_node.send_request_vote("10.0.0.2:50051", term=1, candidate_ip="10.0.0.1:50051")

        assert three_node.state == NodeState.FOLLOWER
        assert three_node.current_term == 10
        assert three_node.voted_for is None

    def test_does_not_step_down_on_equal_term(self, three_node):
        """A peer responding with the same term must NOT cause a step-down."""
        three_node.current_term = 3
        three_node.state = NodeState.CANDIDATE

        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.request_vote.return_value = (True, 3)
            three_node.send_request_vote("10.0.0.2:50051", term=3, candidate_ip="10.0.0.1:50051")

        assert three_node.state == NodeState.CANDIDATE  # Unchanged

    def test_does_not_step_down_on_lower_term(self, three_node):
        """A peer responding with a lower term must NOT affect our state."""
        three_node.current_term = 5
        three_node.state = NodeState.CANDIDATE

        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.request_vote.return_value = (True, 3)
            three_node.send_request_vote("10.0.0.2:50051", term=5, candidate_ip="10.0.0.1:50051")

        assert three_node.state == NodeState.CANDIDATE
        assert three_node.current_term == 5

    def test_client_is_always_closed(self, three_node):
        """The gRPC channel must be closed after every call, even on success."""
        with patch("orchestrator.node.ClusterClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.request_vote.return_value = (True, 1)
            three_node.send_request_vote("10.0.0.2:50051", term=1, candidate_ip="10.0.0.1:50051")

        mock_instance.close.assert_called_once()

    def test_client_is_closed_even_on_exception(self, three_node):
        """
        send_request_vote uses a try/finally — close() must be called even if
        request_vote raises unexpectedly.
        """
        with patch("orchestrator.node.ClusterClient") as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.request_vote.side_effect = RuntimeError("boom")

            with pytest.raises(RuntimeError):
                three_node.send_request_vote("10.0.0.2:50051", term=1, candidate_ip="10.0.0.1:50051")

        mock_instance.close.assert_called_once()



# ---------------------------------------------------------------------------
# broadcast_topology
# ---------------------------------------------------------------------------

class TestBroadcastTopology:
    def test_sets_own_topology_config(self, three_node):
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.broadcast_topology.return_value = (True, 8 * 1024 ** 3)
            three_node.broadcast_topology()

        assert three_node.topology_config is not None

    def test_sets_own_coordinator_ip_to_self(self, three_node):
        """The leader sets itself as coordinator_ip before sending to peers."""
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.broadcast_topology.return_value = (True, 8 * 1024 ** 3)
            three_node.broadcast_topology()

        assert three_node.coordinator_ip == "10.0.0.1:50051"

    def test_topology_config_coordinator_ip(self, three_node):
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.broadcast_topology.return_value = (True, 8 * 1024 ** 3)
            three_node.broadcast_topology()

        assert three_node.topology_config.coordinator_ip == "10.0.0.1:50051"

    def test_ordered_ips_includes_all_nodes(self, three_node):
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.broadcast_topology.return_value = (True, 8 * 1024 ** 3)
            three_node.broadcast_topology()

        ordered = list(three_node.topology_config.ordered_node_ips)
        assert set(ordered) == {"10.0.0.1:50051", "10.0.0.2:50051", "10.0.0.3:50051"}
        assert len(ordered) == 3

    def test_leader_is_first_in_ordered_ips(self, three_node):
        """Leader's own IP must be placed at index 0 of the ordered list."""
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.broadcast_topology.return_value = (True, 8 * 1024 ** 3)
            three_node.broadcast_topology()

        ordered = list(three_node.topology_config.ordered_node_ips)
        assert ordered[0] == "10.0.0.1:50051"

    def test_sends_to_every_peer(self, three_node):
        """One ClusterClient must be instantiated (and closed) per peer."""
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.broadcast_topology.return_value = (True, 8 * 1024 ** 3)
            three_node.broadcast_topology()

        assert MockClient.call_count == len(three_node.peer_ips)  # 2

    def test_peer_ips_used_as_targets(self, three_node):
        """Each peer IP must appear exactly once in the ClusterClient call args."""
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.broadcast_topology.return_value = (True, 8 * 1024 ** 3)
            three_node.broadcast_topology()

        called_ips = {c.kwargs.get("target_ip") or c.args[0] for c in MockClient.call_args_list}
        assert called_ips == {"10.0.0.2:50051", "10.0.0.3:50051"}

    def test_peer_capacities_populated_on_success(self, three_node):
        """Capacities returned by peers must be stored in peer_capacities."""
        with patch("orchestrator.node.ClusterClient") as MockClient:
            MockClient.return_value.broadcast_topology.return_value = (True, 4 * 1024 ** 3)
            three_node.broadcast_topology()

        for peer in three_node.peer_ips:
            assert peer in three_node.peer_capacities
            assert three_node.peer_capacities[peer] == 4 * 1024 ** 3


# ---------------------------------------------------------------------------
# _create_topology_config
# ---------------------------------------------------------------------------

class TestCreateTopologyConfig:
    """Unit tests for the topological graph evaluation static helper."""
    
    def test_head_element(self):
        ordered_ips = ["10.0.0.1:50051", "10.0.0.2:50051", "10.0.0.3:50051", "10.0.0.4:50051"]
        topo = ClusterNode._create_topology_config("10.0.0.1:50051", "10.0.0.1:50051", ordered_ips, 1)
        
        assert topo.node_index == 0
        assert topo.prev_node_idx == -1
        assert topo.next_node_idx == 1

    def test_middle_element(self):
        ordered_ips = ["10.0.0.1:50051", "10.0.0.2:50051", "10.0.0.3:50051", "10.0.0.4:50051"]
        topo = ClusterNode._create_topology_config("10.0.0.2:50051", "10.0.0.1:50051", ordered_ips, 1)
        
        assert topo.node_index == 1
        assert topo.prev_node_idx == 0
        assert topo.next_node_idx == 2

    def test_tail_element(self):
        ordered_ips = ["10.0.0.1:50051", "10.0.0.2:50051", "10.0.0.3:50051", "10.0.0.4:50051"]
        topo = ClusterNode._create_topology_config("10.0.0.4:50051", "10.0.0.1:50051", ordered_ips, 1)
        
        assert topo.node_index == 3
        assert topo.prev_node_idx == 2
        assert topo.next_node_idx == 4

    def test_missing_element_raises_value_error(self):
        """An IP not in ordered_ips must raise ValueError from list.index()."""
        ordered_ips = ["10.0.0.1:50051", "10.0.0.2:50051", "10.0.0.3:50051", "10.0.0.4:50051"]
        with pytest.raises(ValueError):
            ClusterNode._create_topology_config("10.0.0.9:50051", "10.0.0.1:50051", ordered_ips, 1)
