import grpc
from .proto import cluster_service_pb2
from .proto import cluster_service_pb2_grpc


class ClusterClient:
    def __init__(self, target_ip="localhost", port=50051):
        self.target_ip = target_ip if ":" in target_ip else f"{target_ip}:{port}"
        self.port = port
        self.channel = grpc.insecure_channel(self.target_ip)
        self.stub = cluster_service_pb2_grpc.ClusterCoordinatorStub(self.channel)

    def request_vote(self, term: int, candidate_ip: str) -> tuple[bool, int]:
        """
        Sends RequestVote gRPC to the target peer.
        Returns a tuple of (vote_granted: bool, responder_term: int).
        """
        request = cluster_service_pb2.VoteRequest(
            term=term,
            candidate_ip=candidate_ip
        )
        try:
            # 200ms timeout for network call 
            response = self.stub.RequestVote(request, timeout=0.2)
            return response.vote_granted, response.term
        except grpc.RpcError:
            # If the node is unreachable or offline, default to denying vote
            return False, 0

    def broadcast_topology(self, topology: cluster_service_pb2.TopologyConfig) -> tuple[bool, int]:
        """
        Sends BroadcastTopology gRPC to the target peer.
        """
        try:
            # 1.0s timeout for topology dissemination
            response = self.stub.BroadcastTopology(topology, timeout=10.0)
            return response.ok, response.available_memory_bytes
        except grpc.RpcError:
            return False, 0

    def broadcast_partitioning(self, config: cluster_service_pb2.PartitionConfig) -> bool:
        """
        Sends BroadcastPartitioning gRPC to the target peer.
        """
        try:
            response = self.stub.BroadcastPartitioning(config, timeout=10.0)
            return response.ok
        except grpc.RpcError:
            return False

    def ping(self) -> bool:
        """Sends Ping RPC to verify node is online."""
        try:
            request = cluster_service_pb2.PingRequest()
            response = self.stub.Ping(request, timeout=0.5)
            return response.ok
        except grpc.RpcError:
            return False

    def close(self):
        """Cleanly shut down the gRPC channel."""
        self.channel.close()