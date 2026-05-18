import socket
import json
import torch
import threading

def get_vram_gb():
    try:
        if torch.cuda.is_available():
            # Asks the GPU hardware directly: (Total VRAM - Free VRAM)
            free, total = torch.cuda.mem_get_info()
            return (total - free) / (1024 ** 3)
            
        elif torch.backends.mps.is_available():
            # Apple Silicon shares Unified Memory, so we track PyTorch's footprint
            return torch.mps.current_allocated_memory() / (1024 ** 3)
            
        return 0.0
    except Exception:
        return 0.0

class TelemetryClient:
    def __init__(self, target_ip, port=8081, node_id=None):
        self.target_ip = target_ip
        self.port = port
        self.node_id = node_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_event = threading.Event()
        self._thread = None

    def start_heartbeat(self, interval=1.0):
        if self.node_id is None:
            return
        self._thread = threading.Thread(target=self._heartbeat_loop, args=(interval,), daemon=True)
        self._thread.start()

    def _heartbeat_loop(self, interval):
        while not self._stop_event.is_set():
            vram = get_vram_gb()
            self.send_metrics(self.node_id, vram)
            self._stop_event.wait(interval)

    def send_metrics(self, node_id, vram, loss=None, fw_time=None, bw_time=None):
        try:
            payload = {"node_id": node_id, "vram": vram}
            if loss is not None:
                payload["loss"] = loss
            if fw_time is not None:
                payload["fw_time"] = fw_time
            if bw_time is not None:
                payload["bw_time"] = bw_time
                
            msg = json.dumps(payload).encode('utf-8')
            self.sock.sendto(msg, (self.target_ip, self.port))
        except Exception:
            pass