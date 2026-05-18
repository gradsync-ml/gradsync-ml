import inspect
import random
import threading
import time
import grpc
from concurrent.futures import ThreadPoolExecutor, as_completed

from .states import NodeState
from .client import ClusterClient
from .server import ClusterServer
from .proto import cluster_service_pb2, cluster_service_pb2_grpc


class ClusterNode:
    def __init__(self, host_ip: str, peer_ips: list, port: int = 50051):

        self.host_ip = host_ip if ":" in host_ip else f"{host_ip}:{port}"
        self.peer_ips = [p if ":" in p else f"{p}:{port}" for p in peer_ips]
        self.port = port
        self.peer_capacities = {}
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None    # candidate_ip this node voted for in current term
        self.votes_received = 0
        self.coordinator_ip = None
        self.topology_config = None
        self.partition_config = None
        self.server = None
        self._election_cv = threading.Condition()  # used to sleep until timeout (or future heartbeat)

    # ------------------------------------------------------------------
    # State-transition helper
    # ------------------------------------------------------------------
    def _set_state(self, new_state: "NodeState") -> None:
        """Set node state and emit a structured log line with the call-site location.

        The log format is:
            [{host_ip}] STATE {OLD} -> {NEW}  ({file}:{line})
        """
        old_state = self.state
        if old_state == new_state:
            return  # no-op – skip noise for unchanged transitions
        self.state = new_state
        frame = inspect.stack()[1]  # one level up = the actual call site
        location = f"{frame.filename}:{frame.lineno}"
        print(f"[{self.host_ip}] STATE {old_state.name} -> {new_state.name}  ({location})")

    def _serve_cluster(self):
        """Starts the background gRPC server for the Raft node."""
        server = grpc.server(ThreadPoolExecutor(max_workers=10))
        cluster_service_pb2_grpc.add_ClusterCoordinatorServicer_to_server(
            ClusterServer(self), server
        )
        bind_ip = f"0.0.0.0:{self.host_ip.split(":")[-1]}"
        print(f"Starting Orchestrator on {bind_ip}")
        bound_port = server.add_insecure_port(bind_ip)
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind to {self.host_ip}. The port is likely already in use by another process.")
            
        print(f"[{self.host_ip}] Raft Server listening on {self.host_ip}...")
        server.start()
        return server

    def join_cluster(self):
        """
        Main orchestrator lifecycle. Blocks until the cluster agrees on a single
        coordinator (LEADER) and a finalized network topology.
        Returns the finalized topology (TopologyConfig).
        """
        # Start the background gRPC server to receive votes and topology
        self.server = self._serve_cluster()
        
        # Halt execution and wait for all pipeline peers to come online completely
        self.wait_for_peers()

        while self.topology_config is None:
            # Randomized election timeout: 300–450 ms
            election_timeout = random.uniform(0.300, 0.450)

            with self._election_cv:
                notified = self._election_cv.wait(timeout=election_timeout)

                if self.topology_config is not None:
                    break
                
                if notified or self.state == NodeState.LEADER:
                    # Woken up by incoming BroadcastTopology (which sets topology_config),
                    # or we somehow are already leader. Loop condition will handle exiting.
                    continue

                # Timeout fired. Transition to CANDIDATE and start election
                self._set_state(NodeState.CANDIDATE)
                self.current_term += 1
                print(f"[{self.host_ip}] Election timeout fired (term={self.current_term}). Becoming CANDIDATE.")

                self.votes_received = 1
                self.voted_for = self.host_ip

                proposed_term = self.current_term
                candidate_for_vote = self.host_ip

                # Immediately check if majority is met (handles 1-node cluster zero-peer case)
                total_nodes = len(self.peer_ips) + 1
                print(f"[{self.host_ip}] Total nodes: {total_nodes}, Votes received: {self.votes_received}")
                is_leader_now = False
                if self.votes_received > total_nodes // 2:
                    self._set_state(NodeState.LEADER)
                    print(f"[{self.host_ip}] Elected LEADER for term {self.current_term}!")
                    # Set topology immediately under the lock so the outer while-loop
                    # cannot fire another election timeout before we broadcast.
                    ordered_ips = [self.host_ip] + self.peer_ips
                    self.topology_config = self._create_topology_config(
                        self.host_ip, self.host_ip, ordered_ips, self.current_term
                    )
                    self.coordinator_ip = self.host_ip
                    self._election_cv.notify_all()
                    is_leader_now = True

            # --- LOCK RELEASED ---

            if is_leader_now:
                self.broadcast_topology()
                continue

            if self.state != NodeState.CANDIDATE:
                continue
            print(f"[{self.host_ip}] Requesting votes from peers: {self.peer_ips}")
            # Execute vote requests concurrently
            with ThreadPoolExecutor(max_workers=max(1, len(self.peer_ips))) as executor:
                futures = {
                    executor.submit(self.send_request_vote, peer, proposed_term, candidate_for_vote): peer
                    for peer in self.peer_ips
                }

                for future in as_completed(futures):
                    try:
                        vote_granted = future.result()
                        
                        if vote_granted:
                            is_leader_now = False
                            # --- ACQUIRE LOCK BRIEFLY ---
                            with self._election_cv:
                                # If another peer won the election or we started a new term, ignore stale votes
                                if self.state != NodeState.CANDIDATE or self.current_term != proposed_term:
                                    break
                                    
                                self.votes_received += 1
                                
                                # Check majority (N/2 + 1 where N = self + peers)
                                total_nodes = len(self.peer_ips) + 1
                                if self.votes_received > total_nodes // 2:
                                    self._set_state(NodeState.LEADER)
                                    print(f"[{self.host_ip}] Elected LEADER for term {self.current_term}!")
                                    # Set topology immediately under the lock so the outer while-loop
                                    # cannot fire another election timeout before we broadcast.
                                    ordered_ips = [self.host_ip] + self.peer_ips
                                    self.topology_config = self._create_topology_config(
                                        self.host_ip, self.host_ip, ordered_ips, self.current_term
                                    )
                                    self.coordinator_ip = self.host_ip
                                    self._election_cv.notify_all()
                                    is_leader_now = True
                            
                            if is_leader_now:
                                self.broadcast_topology()
                                break

                    except Exception as e:
                        peer_ip = futures[future]
                        print(f"[{self.host_ip}] Failed to get vote from {peer_ip}. Re-trying next cycle.")
        
        return self.topology_config, self.peer_capacities

    def wait_for_partitioning(self):
        """Blocks until the Leader sends the layer partition boundaries."""
        with self._election_cv:
            while self.partition_config is None:
                self._election_cv.wait()
        return self.partition_config

    def shutdown(self):
        """Cleanly stop the gRPC server."""
        if self.server:
            self.server.stop(grace=None)

    def wait_for_peers(self):
        """Blocks indefinitely until all peer IPs return a successful gRPC Ping."""
        if not self.peer_ips:
            return

        print(f"[{self.host_ip}] Waiting for peers to come online: {self.peer_ips}")
        pending = set(self.peer_ips)
        
        while pending:
            # We iterate over a copy (list) so we can safely remove from the original set
            for peer in list(pending):
                client = ClusterClient(target_ip=peer, port=self.port)
                try:
                    if client.ping():
                        print(f"[{self.host_ip}] Peer {peer} is ONLINE!")
                        pending.remove(peer)
                    else:
                        print(f"[{self.host_ip}] Peer {peer} ping returned False.")
                except Exception as e:
                    print(f"[{self.host_ip}] Exception pinging {peer}: {e}")
                finally:
                    client.close()
                    
            if pending:
                time.sleep(0.5)

    def send_request_vote(self, peer_ip: str, term: int, candidate_ip: str) -> bool:
        """Sends RequestVote gRPC to peer_ip."""
        client = ClusterClient(target_ip=peer_ip, port=self.port)
        try:
            vote_granted, responder_term = client.request_vote(term, candidate_ip)

            # Standard Raft rule: If a peer responds with a term greater than ours, 
            # we are out-of-date and must immediately revert to FOLLOWER
            with self._election_cv:
                if responder_term > self.current_term:
                    self.current_term = responder_term
                    self._set_state(NodeState.FOLLOWER)
                    self.voted_for = None
                    print(f"[{self.host_ip}] Peer {peer_ip} has higher term {responder_term}. Stepping down to FOLLOWER.")

            return vote_granted
        finally:
            client.close()

    def broadcast_topology(self):
        """Called by the newly elected LEADER to tell all peers the final assignments.

        topology_config is already set on the leader by join_cluster() under the lock
        before this method is called. This method only fans out to ALL peers and
        waits for every acknowledgment, since every node must have the topology for
        the cluster to be usable.
        """
        ordered_ips = [self.host_ip] + self.peer_ips

        # --- Broadcast to ALL peers concurrently and wait for every ack ---
        with ThreadPoolExecutor(max_workers=max(1, len(self.peer_ips))) as executor:
            futures = {
                executor.submit(self.send_topology, peer, self.host_ip, ordered_ips, self.current_term): peer
                for peer in self.peer_ips
            }

            for future in as_completed(futures):
                peer = futures[future]
                try:
                    ok, capacity = future.result()
                    if ok:
                        self.peer_capacities[peer] = capacity
                        print(f"[{self.host_ip}] Topology ack received from {peer}.")
                    else:
                        print(f"[{self.host_ip}] Peer {peer} rejected topology broadcast.")
                except Exception as e:
                    print(f"[{self.host_ip}] Failed to send topology to {peer}: {e}")

    def send_topology(self, peer_ip: str, coordinator_ip: str, ordered_ips: list[str], term: int) -> tuple[bool, int]:
        """Sends BroadcastTopology to peer_ip."""
        client = ClusterClient(target_ip=peer_ip, port=self.port)
        try:
            topology = self._create_topology_config(peer_ip, coordinator_ip, ordered_ips, term)
            return client.broadcast_topology(topology)
        finally:
            client.close()

    @staticmethod
    def _create_topology_config(target_ip: str, coordinator_ip: str, ordered_ips: list[str], term: int):
        """Helper to generate a properly indexed TopologyConfig payload for a specific target node."""
        # try:
        idx = ordered_ips.index(target_ip)
        prev_idx = idx - 1
        next_idx = idx + 1
        # except ValueError:
        #     idx, prev_idx, next_idx = -1, 

        return cluster_service_pb2.TopologyConfig(
            coordinator_ip=coordinator_ip,
            ordered_node_ips=ordered_ips,
            term=term,
            node_index=idx,
            prev_node_idx=prev_idx,
            next_node_idx=next_idx
        )

    def broadcast_partitioning(self, allocations: dict):
        """Called by LEADER to send specific layer boundaries to each node."""
        if self.state != NodeState.LEADER:
            return
            
        with self._election_cv:
            # Set our own config
            my_alloc = allocations.get(self.host_ip)
            if my_alloc:
                self.partition_config = cluster_service_pb2.PartitionConfig(
                    start_layer_idx=my_alloc['start'],
                    end_layer_idx=my_alloc['end']
                )

        with ThreadPoolExecutor(max_workers=max(1, len(self.peer_ips))) as executor:
            for peer in self.peer_ips:
                if peer in allocations:
                    executor.submit(self.send_partitioning, peer, allocations[peer])

    def send_partitioning(self, peer_ip: str, alloc: dict) -> bool:
        client = ClusterClient(target_ip=peer_ip, port=self.port)
        try:
            config = cluster_service_pb2.PartitionConfig(
                start_layer_idx=alloc['start'],
                end_layer_idx=alloc['end']
            )
            return client.broadcast_partitioning(config)
        finally:
            client.close()