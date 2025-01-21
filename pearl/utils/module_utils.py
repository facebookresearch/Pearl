# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


from typing import Any, Dict

import torch
import torch.nn as nn


StateDictOwner = nn.Module | torch.optim.Optimizer | Dict[str, Any]


def state_dict_owners_have_similar_state_dict(
    sdo1: StateDictOwner,
    sdo2: StateDictOwner,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> str:
    """
    Checks if two PyTorch state dict owners have approximately the same state_dict,
    handling nested state dict owners recursively.

    Args:
      sdo1: The first PyTorch state dict owner.
      sdo2: The second PyTorch state dict owner.
      rtol (float): The relative tolerance.
      atol (float): The absolute tolerance.

    Returns:
      str: Difference description or "" if the state dict owners have similar state_dict.
    """

    sd1: Dict[str, Any] = sdo1 if isinstance(sdo1, dict) else sdo1.state_dict()
    sd2: Dict[str, Any] = sdo2 if isinstance(sdo2, dict) else sdo2.state_dict()

    if set(sd1.keys()) != set(sd2.keys()):
        return (
            f"State dicts have differing keys: {set(sd1.keys())} vs {set(sd2.keys())}, "
            + f"difference: {set(sd1.keys()) ^ set(sd2.keys())}"
        )

    for key in sd1:
        assert (value_is_dict := isinstance(sd1[key], dict)) == isinstance(
            sd2[key], dict
        ), f"{key} must be dict in both state dict owners being compared, or in neither"
        if value_is_dict:  # Check for nested state dicts
            if reason := modules_have_similar_state_dict(
                sd1[key], sd2[key], rtol, atol
            ):
                return reason
        elif isinstance(sd1[key], torch.Tensor) and isinstance(sd2[key], torch.Tensor):
            if not torch.allclose(sd1[key], sd2[key], rtol=rtol, atol=atol):
                return f"Key {key} has different values: {sd1[key]} vs {sd2[key]}"
        elif sd1[key] != sd2[key]:
            return f"Key {key} has different values: {sd1[key]} vs {sd2[key]}"

    return ""


def modules_have_similar_state_dict(
    module1: nn.Module, module2: nn.Module, rtol: float = 1e-05, atol: float = 1e-08
) -> str:
    """
    Checks if two PyTorch modules have approximately the same state_dict,
    handling nested modules recursively.

    Args:
      module1: The first PyTorch module.
      module2: The second PyTorch module.
      rtol (float): The relative tolerance.
      atol (float): The absolute tolerance.

    Returns:
      str: Difference description or "" if the modules have similar state_dict.
    """
    return state_dict_owners_have_similar_state_dict(module1, module2, rtol, atol)


def optimizers_have_similar_state_dict(
    optimizer1: torch.optim.Optimizer,
    optimizer2: torch.optim.Optimizer,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> str:
    """
    Checks if two PyTorch optimizers have approximately the same state_dict,
    handling nested optimizers recursively.

    Args:
      optimizer1: The first PyTorch optimizer.
      optimizer2: The second PyTorch optimizer.
      rtol (float): The relative tolerance.
      atol (float): The absolute tolerance.

    Returns:
      str: Difference description or "" if the optimizers have similar state_dict.
    """
    return state_dict_owners_have_similar_state_dict(optimizer1, optimizer2, rtol, atol)
