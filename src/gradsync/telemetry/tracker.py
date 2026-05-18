import asyncio
import json

class TelemetryTracker:
    def __init__(self):
        self.metrics = {
            "loss": 0.0,
            "vram": {},
            "fw_time": {}, # New
            "bw_time": {}, # New
            "step": 0
        }
        self.clients = set()

    async def update(self, payload):
        node_id = str(payload.get("node_id", "unknown"))
        
        if "vram" in payload:
            self.metrics["vram"][node_id] = payload["vram"]
        if "fw_time" in payload:
            self.metrics["fw_time"][node_id] = payload["fw_time"]
        if "bw_time" in payload:
            self.metrics["bw_time"][node_id] = payload["bw_time"]
        if "loss" in payload:
            self.metrics["loss"] = payload["loss"]
            
        self.metrics["step"] += 1
        await self.broadcast()

    async def broadcast(self):
        if not self.clients: return
        dead_clients = set()
        for client in self.clients:
            try:
                await client.send_json(self.metrics)
            except Exception:
                dead_clients.add(client)
        self.clients -= dead_clients

tracker = TelemetryTracker()

class UDPTelemetryProtocol(asyncio.DatagramProtocol):
    def datagram_received(self, data, addr):
        try:
            payload = json.loads(data.decode('utf-8'))
            loop = asyncio.get_running_loop()
            loop.create_task(tracker.update(payload))
        except Exception:
            pass