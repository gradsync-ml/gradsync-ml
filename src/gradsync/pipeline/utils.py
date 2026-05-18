"""
Utilities for device detection and configuration in distributed pipeline training.
"""

import torch
import struct


import numpy as np
import logging

# Import from your new compression lab
from gradsync.compression_lab import (
    TensorCompressor, 
    CompressionType, 
    get_optimal_compression
)

logging.basicConfig(
    level=logging.INFO,
    # Added [Thread: %(threadName)s] to the format
    format='%(asctime)s | %(levelname)-8s | %(name)s | [Thread: %(threadName)s] | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- MASTER OVERRIDE SWITCH ---
# Set to None to use the intelligent dynamic router.
# Set to CompressionType.NONE to completely disable compression (Raw FP32).
FORCE_COMPRESSION = None
# ------------------------------

COMP_TO_ID = {
    CompressionType.NONE: 0,
    CompressionType.FP16: 1,
    CompressionType.INT8: 2,
    CompressionType.BINARY: 3,
    CompressionType.SPARSE: 4,
    CompressionType.OUTLIER_INT8: 5,
    CompressionType.OUTLIER_INT4: 6
}
ID_TO_COMP = {v: k for k, v in COMP_TO_ID.items()}


def pack_tensor(t: torch.Tensor):
    """
    Compresses a tensor and prepends a 1-byte routing header.
    Supports dynamic routing or forced overrides.
    """
    shape = list(t.shape)

    # 1. Determine Compression Strategy
    if FORCE_COMPRESSION is not None:
        active_compression = FORCE_COMPRESSION
    else:
        active_compression = get_optimal_compression(t)
        # logger.info(f"Active compression: {active_compression.name}")
    # 2. Convert PyTorch tensor to raw FP32 bytes (This is the baseline)
    tensor_bytes = t.detach().cpu().float().numpy().tobytes()

    # 3. Compress (If CompressionType.NONE, this just returns tensor_bytes instantly)
    compressor = TensorCompressor()
    compressed_bytes = compressor.compress(tensor_bytes, compression_type=active_compression)
    
    # 4. Prepend the 1-byte header
    comp_id = COMP_TO_ID[active_compression]
    payload_with_header = struct.pack('!B', comp_id) + compressed_bytes    

    return payload_with_header, shape


def unpack_tensor(payload: bytes, shape, device):
    """
    Reads the 1-byte header, routes to the correct decompressor, and restores the tensor.
    """
    # 1. Extract the 1-byte header
    comp_id = struct.unpack('!B', payload[:1])[0]
    active_compression = ID_TO_COMP[comp_id]

    # 2. Strip the header
    compressed_bytes = payload[1:]

    # 3. Decompress (If CompressionType.NONE, this does nothing)
    compressor = TensorCompressor()
    decompressed_bytes = compressor.decompress(compressed_bytes, active_compression.value)

    # 4. Convert raw FP32 bytes back to PyTorch tensor
    t_np = np.frombuffer(decompressed_bytes, dtype=np.float32)
    t_tensor = torch.frombuffer(t_np.copy(), dtype=torch.float32).reshape(shape)

    return t_tensor.to(device)


def detect_device() -> torch.device:
    """
    Detect the optimal device for a given node role in distributed training.

    Args:
        role: The node role - 'head' for the head node, 'tail' for the tail node

    Returns:
        torch.device: The detected device for the given role

    Raises:
        ValueError: If role is not 'head' or 'tail'
    """

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        # Head node: prefer MPS on Apple Silicon, fallback to CPU
        device = "mps"
    else:
        device = "cpu"

    print("Device is:", device)
    return torch.device(device)


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed information about a device for logging/debugging.

    Args:
        device: The torch device to inspect

    Returns:
        dict: Device information including name, type, and capabilities
    """
    device_info = {
        "device": str(device),
        "type": device.type,
        "index": device.index,
    }

    if device.type == "cuda":
        device_info.update({
            "name": torch.cuda.get_device_name(device),
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
        })

    return device_info
