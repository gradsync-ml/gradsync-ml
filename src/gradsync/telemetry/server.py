from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import threading
from .tracker import tracker, UDPTelemetryProtocol
from .dashboard import HTML_CONTENT

app = FastAPI()

@app.get("/")
async def get():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tracker.clients.add(websocket)
    try:
        while True: 
            await websocket.receive_text()
    except Exception:
        tracker.clients.remove(websocket)

def _run_server(port, udp_port):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="critical")
    server = uvicorn.Server(config)
    
    # Boot UDP Listener
    loop.run_until_complete(
        loop.create_datagram_endpoint(
            lambda: UDPTelemetryProtocol(),
            local_addr=("0.0.0.0", udp_port)
        )
    )
    
    # Boot Web Server
    loop.run_until_complete(server.serve())

def start_telemetry_server(port=8080, udp_port=8081):
    t = threading.Thread(target=_run_server, args=(port, udp_port), daemon=True)
    t.start()