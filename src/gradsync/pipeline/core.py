import sys
import torch
import torch.nn as nn
from .runner import HeadNodeRunner, TailNodeRunner, MiddleNodeRunner
from .utils import detect_device
from gradsync.orchestrator.node import ClusterNode
import time
import json
import asyncio
from gradsync.common.hardware import get_available_memory

from gradsync.telemetry.server import start_telemetry_server


def _ensure_unique_endpoints(endpoints: list[str], field_name: str) -> None:
    duplicates = sorted({endpoint for endpoint in endpoints if endpoints.count(endpoint) > 1})
    if duplicates:
        raise ValueError(f"Invalid config: duplicate entries in {field_name}: {duplicates}")

def validate_cluster_config(
    election_nodes: list[str],
    cluster_nodes: list[str],
    election_host_address: str,
    host_address: str,
) -> None:
    if len(election_nodes) != len(cluster_nodes):
        raise ValueError(
            "Invalid config: election_nodes and cluster_nodes must have the same length "
            f"(got {len(election_nodes)} and {len(cluster_nodes)})"
        )

    _ensure_unique_endpoints(election_nodes, "election_nodes")
    _ensure_unique_endpoints(cluster_nodes, "cluster_nodes")

    if election_host_address not in election_nodes:
        raise ValueError(
            f"Invalid config: local election endpoint {election_host_address} is not listed in election_nodes"
        )

    if host_address not in cluster_nodes:
        raise ValueError(
            f"Invalid config: local data endpoint {host_address} is not listed in cluster_nodes"
        )


class DistributedPipeline(nn.Module):
    def __init__(self, model_builder, criterion: nn.Module, optim_class, optim_kwargs: dict, host_ip: str, elec_port: str, train_port: str, config_path: str):
        super().__init__()

        self.criterion = criterion
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find {config_path}. Please create it!")
            sys.exit(1)
        
        self.host_address = f"{host_ip}:{train_port}"
        self.election_host_address = f"{host_ip}:{elec_port}"
        
        raw_election = config.get("election_nodes", [])
       
        print(f"Election Nodes: {raw_election}")
        

        raw_cluster = config.get("cluster_nodes", [])
        print(f"Cluster Nodes: {raw_cluster}")
        validate_cluster_config(
            raw_election,
            raw_cluster,
            self.election_host_address,
            self.host_address,
        )

        self.election_addresses = [addr for addr in raw_election if addr != self.election_host_address]
        self.peer_addresses = [addr for addr in raw_cluster if addr != self.host_address]
        self.n_micro = config.get("n_micro", 4)
        
        self.elec_to_data_map = dict(zip(raw_election, raw_cluster))
    
        print(f"Peers: {self.peer_addresses} | Micro-batches: {self.n_micro}")
        
        self.local_ip, self.local_port_str = self.host_address.split(':')
        self.local_port = int(self.local_port_str)
        
        self.device = detect_device()
        self.role = None
        self.runner = None

        layer_memory_reqs = self._meta_profile(model_builder)

        self.join_cluster(model_builder, layer_memory_reqs)

        params = list(self.parameters())
        if params:
            self.runner.optimizer = self.optim_class(params, **self.optim_kwargs)
        else:
            print(f"[{self.host_address}] No layers assigned. Operating as pass-through relay.")
            self.runner.optimizer = None
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.start()

    def _meta_profile(self, model_builder):
        with torch.device('meta'):
            meta_model = model_builder()
        
        layer_memory_reqs = []
        for layer in meta_model.layers:
            layer_params = sum(p.numel() * p.element_size() for p in layer.parameters())
            # x4 rough estimate for gradients and optimizer states
            layer_memory_reqs.append(layer_params * 4)
            
        return layer_memory_reqs

    def join_cluster(self, model_builder, layer_memory_reqs):
        print(f"[{self.host_address}] Initiating Cluster Election...")
        node = ClusterNode(host_ip=self.election_host_address, peer_ips=self.election_addresses)
        topology, capacities = node.join_cluster()
        print(capacities)

        idx = topology.node_index
        total_nodes = len(topology.ordered_node_ips)
        total_layers = len(layer_memory_reqs)

        self.pipeline_depth = total_nodes

        if idx == 0:
            self.role = 'head'
            # Leader computes partition boundaries
            allocations = {}
            current_layer = 0
            memory_ip = []
            for ip in topology.ordered_node_ips:
                start_idx = current_layer
                
                if ip not in capacities and ip == self.election_host_address:
                    cap = get_available_memory()
                else:
                    # FIX: Use a safe default (e.g., 8GB) instead of float('inf') to prevent NaN tensors
                    cap = capacities.get(ip, 100 * 1024**3) 

                memory_ip.append((ip, cap))

            ratios = torch.tensor([cap for _, cap in memory_ip])
            ratios = ratios / ratios.sum()
            layer_memory_reqs_tensor = torch.tensor(layer_memory_reqs)
            cumulative_memory = torch.cumsum(layer_memory_reqs_tensor, dim=0)
            mem_alloc = ratios * cumulative_memory[-1]  
            mem_alloc = torch.cumsum(mem_alloc, dim=0)

            print(ratios, mem_alloc, cumulative_memory)
            for i in range(len(topology.ordered_node_ips)):  
                ip = topology.ordered_node_ips[i]
                start_idx = current_layer
                
                while current_layer <= total_layers and cumulative_memory[current_layer] < mem_alloc[i]:
                    current_layer += 1
                
                if current_layer == total_layers - 1:
                    current_layer += 1
                
                current_layer = max(current_layer, start_idx + 1)

                allocations[ip] = {'start': start_idx, 'end': current_layer}
            print(f"Layer Allocations: {allocations}")
            
            try:
                assert current_layer == total_layers
            except AssertionError:
                raise AssertionError(f"FATAL: Cluster Out of Memory! Could only fit {current_layer}/{total_layers} layers.")

            node.broadcast_partitioning(allocations)
            partition_config = node.partition_config
        else:
            if idx == total_nodes - 1:
                self.role = 'tail'
            else:
                self.role = 'middle'
            
            print(f"[{self.host_address}] Waiting for leader to partition layers...")
            partition_config = node.wait_for_partitioning()

        start_layer = partition_config.start_layer_idx
        end_layer = partition_config.end_layer_idx

        real_model = model_builder()
        layers = list(real_model.layers)
        my_slice = layers[start_layer:end_layer]

        # Parse next node details
        next_ip, next_port = None, None
        if topology.next_node_idx >= 0 and topology.next_node_idx < len(topology.ordered_node_ips):
            next_elec_ip = topology.ordered_node_ips[topology.next_node_idx]
            next_data_ip = self.elec_to_data_map[next_elec_ip]
            next_ip, next_port_str = next_data_ip.split(':')
            next_port = int(next_port_str)

        telemetry_ip = topology.ordered_node_ips[0] if topology.ordered_node_ips else topology.coordinator_ip

        if self.role == 'head':
            self.runner = HeadNodeRunner(my_slice, target_ip=next_ip, port=next_port, device=self.device, coordinator_ip=topology.coordinator_ip, node_id=idx, telemetry_ip=telemetry_ip)
            start_telemetry_server(port=8080, udp_port=8081)  # Start telemetry server for head node
        elif self.role == 'tail':
            if hasattr(real_model, 'output_layer'):
                my_slice.append(real_model.output_layer)
            self.runner = TailNodeRunner(my_slice, device=self.device, criterion=self.criterion, n_micro=self.n_micro, coordinator_ip=topology.coordinator_ip, node_id=idx, telemetry_ip=telemetry_ip)
            self.serve_port = self.local_port 
        else:
            self.runner = MiddleNodeRunner(my_slice, target_ip=next_ip, port=next_port, device=self.device, n_micro=self.n_micro, coordinator_ip=topology.coordinator_ip, node_id=idx, telemetry_ip=telemetry_ip)
            self.serve_port = self.local_port

        print(f"[{self.host_address}] Assigned Role: {self.role.upper()} | Layers: {start_layer} to {end_layer}")

    def serve_forever(self, port):
        return self.runner.serve(port=port)

    async def train_step(self, inputs, targets):
        if self.role != 'head':
            raise RuntimeError("Only the 'head' node can initiate a train_step.")
        return await self.runner.train_batch(inputs, targets)

    def parameters(self, recurse: bool = True):
        return self.runner.model_slice.parameters(recurse)

    def zero_grad(self):
        if self.role == 'head':
            self.runner.optimizer.zero_grad()

    def step(self):
        if self.role == 'head':
            self.runner.optimizer.step()

    def start(self):
        if self.role in ['middle', 'tail']:
            print(f"[{self.host_address}] Worker Node activated. Serving on port {self.local_port}...")
            
            try:
                self.serve_forever(port=self.local_port)
                
            except KeyboardInterrupt:
                print(f"\n[{self.host_address}] Shutting down worker node cleanly...")
                sys.exit(0)
            
        elif self.role == 'head':
            print(f"[{self.host_address}] Head Node activated. Cluster linked! Ready for training.")

    def execute_batch(self, inputs, targets):
        if self.role != 'head':
            raise RuntimeError("Only the head node can execute training batches.")
        return self._loop.run_until_complete(self._async_execute_batch(inputs, targets))

    async def _async_execute_batch(self, inputs, targets):
        micro_x = torch.chunk(inputs, chunks=self.n_micro, dim=0)
        micro_y = torch.chunk(targets, chunks=self.n_micro, dim=0)

        self.zero_grad()

                # Dynamically limit concurrency based on the active cluster size
        # If 4 laptops join the cluster, this automatically allows 4 concurrent micro-batches
        semaphore = asyncio.Semaphore(self.pipeline_depth)


        async def bounded_train_step(mx, my):
            # Wait for a previous batch to clear the node before sending the next
            async with semaphore:
                return await self.train_step(mx, my)

        # Launch the bounded tasks
        tasks = [bounded_train_step(mx, my) for mx, my in zip(micro_x, micro_y)]
        micro_losses = await asyncio.gather(*tasks)

        self.step()

        return sum(micro_losses) / len(micro_losses)
