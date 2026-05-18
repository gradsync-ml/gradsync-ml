import torch
import torch.nn as nn
from gradsync.comms.server import serve_pipeline
from gradsync.comms.client import PipelineClient
from .utils import pack_tensor, unpack_tensor
import asyncio
import time
import threading

from gradsync.telemetry.client import TelemetryClient, get_vram_gb

class BuddyNode(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.local_layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.local_layers:
            x = layer(x)
        return x


def _resolve_telemetry_target(telemetry_ip=None, coordinator_ip=None):
    target = telemetry_ip or coordinator_ip or "127.0.0.1"
    print("Target ip:", target)
    return target.split(":")[0]


class TailNodeRunner:
    def __init__(self, model_slice_layers, device, criterion, n_micro=1, coordinator_ip=None, node_id=None, telemetry_ip=None):
        self.device = device
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.criterion = criterion
        self.optimizer = None
        self.accum_steps = n_micro
        self.fw_started = 0
        self.bw_completed = 0
        self.lock = threading.Lock()

        # --- INITIALIZE TELEMETRY CLIENT ---
        self.node_id = node_id
        clean_ip = _resolve_telemetry_target(telemetry_ip, coordinator_ip)
        print(f"TailNodeRunner initializing telemetry client targeting {clean_ip}:8081 with node_id {node_id}")
        self.telemetry = TelemetryClient(target_ip=clean_ip, port=8081, node_id=node_id)

        self.telemetry.start_heartbeat(interval=1.0)


    def _process_batch_callback(self, act_bytes, act_shape, tgt_bytes, tgt_shape):
        with self.lock:
            if self.fw_started % self.accum_steps == 0:
                self.optimizer.zero_grad()
            self.fw_started += 1

            activations = unpack_tensor(act_bytes, act_shape, self.device)
            activations.requires_grad_(True)
            
            targets = unpack_tensor(tgt_bytes, tgt_shape, self.device)
            if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                targets = targets.long()

            # ...
            t0 = time.perf_counter()
            outputs = self.model_slice(activations)
            loss = self.criterion(outputs, targets)
            scaled_loss = loss / self.accum_steps
            fw_time = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            scaled_loss.backward()
            bw_time = (time.perf_counter() - t1) * 1000

            self.bw_completed += 1
            if self.bw_completed % self.accum_steps == 0:
                self.optimizer.step()

            grad_bytes, grad_shape = pack_tensor(activations.grad)
            
            vram = get_vram_gb()
            self.telemetry.send_metrics(self.node_id, vram, fw_time=fw_time, bw_time=bw_time)

            return grad_bytes, grad_shape, loss.item()

    def serve(self, port=12345):
        print(f"Tail Node Engine ready. Listening on port {port} (Device: {self.device})...")
        return serve_pipeline(processing_callback=self._process_batch_callback, port=port)

class HeadNodeRunner:
    def __init__(self, model_slice_layers, target_ip, port=12345, device=None, coordinator_ip=None, node_id=None, telemetry_ip=None):
        self.device = device or torch.device("cpu")
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.optimizer = None
        self.client = PipelineClient(target_ip=target_ip, port=port)

        # --- INITIALIZE TELEMETRY CLIENT ---
        self.node_id = node_id
        clean_ip = "127.0.0.1"
        print(f"HeadNodeRunner initializing telemetry client targeting {clean_ip}:8081 with node_id {node_id}")
        self.telemetry = TelemetryClient(target_ip=clean_ip, port=8081, node_id=node_id)

        self.telemetry.start_heartbeat(interval=1.0)

        self.torch_lock = asyncio.Lock()

    async def configure_remote(self, start_layer, end_layer):
        return await self.client.send_pipeline_config(start_layer, end_layer, is_tail=True)

    async def train_batch(self, inputs, targets):
        inputs = inputs.to(self.device)
        
        # 1. FORWARD PASS (Lock it)
        async with self.torch_lock:
            t0 = time.perf_counter()
            local_activations = self.model_slice(inputs)
            fw_time = (time.perf_counter() - t0) * 1000

            act_bytes, act_shape = pack_tensor(local_activations)
            tgt_bytes, tgt_shape = pack_tensor(targets)

        # 2. NETWORK I/O (Unlocked)
        grad_bytes, grad_shape, loss_val = await self.client.send_forward_receive_backward(
            act_bytes, act_shape, tgt_bytes, tgt_shape
        )
        
        # 3. BACKWARD PASS (Lock it)
        async with self.torch_lock:
            returned_grads = unpack_tensor(grad_bytes, grad_shape, self.device)

            t1 = time.perf_counter()
            local_activations.backward(returned_grads)
            bw_time = (time.perf_counter() - t1) * 1000

        vram = get_vram_gb()
        self.telemetry.send_metrics(self.node_id, vram, loss=loss_val, fw_time=fw_time, bw_time=bw_time)

        return loss_val
    
class MiddleNodeRunner:
    def __init__(self, model_slice_layers, target_ip, port, device, n_micro=1, coordinator_ip=None, node_id=None, telemetry_ip=None):
        self.device = device
        self.model_slice = BuddyNode(model_slice_layers).to(self.device)
        self.optimizer = None
        self.client = PipelineClient(target_ip=target_ip, port=port)

        self.accum_steps = n_micro
        self.fw_started = 0
        self.bw_completed = 0
        self.mb_counter = 0

        # --- INITIALIZE TELEMETRY CLIENT ---
        self.node_id = node_id
        clean_ip = _resolve_telemetry_target(telemetry_ip, coordinator_ip)
        print(f"MiddleNodeRunner initializing telemetry client targeting {clean_ip}:8081 with node_id {node_id}")
        self.telemetry = TelemetryClient(target_ip=clean_ip, port=8081, node_id=node_id)

        self.telemetry.start_heartbeat(interval=1.0)
        self.torch_lock = asyncio.Lock()

    async def _process_batch_callback(self, act_bytes, act_shape, tgt_bytes, tgt_shape):
        self.mb_counter += 1
        mb_id = self.mb_counter

        # 1. FORWARD PASS (Compute-bound -> Lock it)
        async with self.torch_lock:
            if self.fw_started % self.accum_steps == 0:
                self.optimizer.zero_grad()
            self.fw_started += 1

            print(f"  [MB {mb_id}] FORWARD Started")

            activations = unpack_tensor(act_bytes, act_shape, self.device)
            activations.requires_grad_(True)
            
            t0 = time.perf_counter()
            local_output = self.model_slice(activations)
            fw_time = (time.perf_counter() - t0) * 1000

            next_act_bytes, next_act_shape = pack_tensor(local_output)
        
        # 2. NETWORK I/O (I/O-bound -> Leave unlocked to allow pipeline overlap)
        grad_bytes, grad_shape, loss_val = await self.client.send_forward_receive_backward(
            next_act_bytes, next_act_shape, tgt_bytes, tgt_shape
        )
        
        # 3. BACKWARD PASS (Compute-bound -> Lock it)
        async with self.torch_lock:
            remote_grads = unpack_tensor(grad_bytes, grad_shape, self.device)
            
            t1 = time.perf_counter()
            local_output.backward(remote_grads)
            bw_time = (time.perf_counter() - t1) * 1000

            self.bw_completed += 1
            if self.bw_completed % self.accum_steps == 0:
                self.optimizer.step()
            
            my_grad_bytes, my_grad_shape = pack_tensor(activations.grad)
        
        vram = get_vram_gb()
        self.telemetry.send_metrics(self.node_id, vram, fw_time=fw_time, bw_time=bw_time)

        return my_grad_bytes, my_grad_shape, loss_val

    def serve(self, port=12345):
        return serve_pipeline(processing_callback=self._process_batch_callback, port=port)
