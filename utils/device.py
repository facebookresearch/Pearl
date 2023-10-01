import torch
import torch.distributed as dist


# pyre-fixme[3]: Return type must be annotated.
def get_pearl_device():
    try:
        # This is to pytorch distributed run, and should not affect
        # original implementation of this file
        local_rank = dist.get_rank()
    except Exception:
        local_rank = 0

    return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


# pyre-fixme[3]: Return type must be annotated.
def is_distribution_enabled():
    return dist.is_initialized() and dist.is_available()
