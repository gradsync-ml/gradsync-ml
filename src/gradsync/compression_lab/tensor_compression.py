import torch

def compress_tensor(tensor: torch.Tensor, target_dtype=torch.float16) -> bytes:
    """Downcasts tensor to reduce network payload size."""
    # Convert to FP16 or INT8 before sending across wire
    compressed = tensor.to(target_dtype)
    return compressed.cpu().detach().numpy().tobytes()

def decompress_tensor(raw_bytes: bytes, shape: tuple, source_dtype=torch.float16) -> torch.Tensor:
    """Reconstructs the tensor on the receiving node."""
    tensor = torch.frombuffer(raw_bytes, dtype=source_dtype).reshape(shape).clone()
    # Cast back to FP32 for standard PyTorch autograd computation
    return tensor.to(torch.float32)