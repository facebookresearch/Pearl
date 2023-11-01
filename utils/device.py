import torch
import torch.distributed as dist


def get_pearl_device(device_id: int = -1) -> torch.device:
    if device_id != -1:
        return torch.device("cuda:" + str(device_id))

    try:
        # This is to pytorch distributed run, and should not affect
        # original implementation of this file
        local_rank = dist.get_rank()
    except Exception:
        local_rank = 1

    return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


def is_distribution_enabled() -> bool:
    return dist.is_initialized() and dist.is_available()
