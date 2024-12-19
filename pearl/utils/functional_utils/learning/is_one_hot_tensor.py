# pyre-unsafe
import torch


def is_one_hot_tensor(tensor: torch.Tensor, dim: int = -1):
    # Ensure tensor is binary (contains only 0s and 1s)
    is_binary = torch.all((tensor == 0) | (tensor == 1))
    if not is_binary:
        return False

    # Check if along the specified dimension, there is exactly one 1 per row
    one_hot_check = torch.sum(tensor, dim=dim) == 1
    return torch.all(one_hot_check)
