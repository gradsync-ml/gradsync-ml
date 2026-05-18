import psutil
import torch

def get_available_memory() -> int:
    """
    Returns the dynamically measured available memory bytes for the local node.
    Prioritizes PyTorch GPU VRAM if CUDA is available, otherwise falls back
    to system CPU RAM via psutil (which is also accurate for Apple Unified Memory).
    """
    if torch.cuda.is_available():
        free_vram, _ = torch.cuda.mem_get_info()
        return free_vram
    else:
        return psutil.virtual_memory().available
