import os

import torch

pearl_device = None


def get_pearl_device():
    global pearl_device
    if pearl_device is None:
        try:
            # This is to pytorch distributed run, and should not affect
            # original implementation of this file
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            pearl_device = (
                torch.device(f"cuda:{local_rank}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            return pearl_device
        except TypeError:
            pass
        pearl_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return pearl_device
