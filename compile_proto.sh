#!/usr/bin/env bash
set -euo pipefail

# Ensure we are running from the root of the project
cd "$(dirname "$0")"

case "$(uname -s 2>/dev/null || echo unknown)" in
  Darwin*) platform="macOS" ;;
  Linux*) platform="Linux" ;;
  MINGW*|MSYS*|CYGWIN*) platform="Windows POSIX shell" ;;
  *) platform="unknown POSIX shell" ;;
esac

echo "Generating protobuf stubs on ${platform}..."

# 1. Compile Comms Protos
echo "Compiling tensor_service.proto..."
mkdir -p src/gradsync/comms/proto
uv run python -m grpc_tools.protoc \
  -I src/gradsync/comms/proto \
  --python_out=src/gradsync/comms/proto \
  --grpc_python_out=src/gradsync/comms/proto \
  src/gradsync/comms/proto/tensor_service.proto

# 2. Compile Orchestrator Protos
echo "Compiling cluster_service.proto..."
mkdir -p src/gradsync/orchestrator/proto
uv run python -m grpc_tools.protoc \
  -I src/gradsync/orchestrator/proto \
  --python_out=src/gradsync/orchestrator/proto \
  --grpc_python_out=src/gradsync/orchestrator/proto \
  src/gradsync/orchestrator/proto/cluster_service.proto

# 3. Patch the gRPC relative import bug
echo "Patching Python relative imports..."
uv run python - <<'PY'
from pathlib import Path

patches = {
    Path("src/gradsync/comms/proto/tensor_service_pb2_grpc.py"): (
        "import tensor_service_pb2 as",
        "from . import tensor_service_pb2 as",
    ),
    Path("src/gradsync/orchestrator/proto/cluster_service_pb2_grpc.py"): (
        "import cluster_service_pb2 as",
        "from . import cluster_service_pb2 as",
    ),
}

for path, (old, new) in patches.items():
    if path.exists():
        text = path.read_text()
        path.write_text(text.replace(old, new))
    else:
        print(f"Warning: {path} not found for patching.")
PY

echo "Generated protobuf stubs successfully."