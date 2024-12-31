# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


from typing import Any, Dict

import torch
import torch.nn as nn


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

    sd1: Dict[str, Any] = module1.state_dict()
    sd2: Dict[str, Any] = module2.state_dict()

    if set(sd1.keys()) != set(sd2.keys()):
        return (
            f"State dicts have differing keys: {set(sd1.keys())} vs {set(sd2.keys())}, "
            + "difference: {set(sd1.keys()) ^ set(sd2.keys())}"
        )

    for key in sd1:
        assert (value_is_dict := isinstance(sd1[key], dict)) == isinstance(
            sd2[key], dict
        ), f"{key} must be dict in both modules being compared, or in neither"
        if value_is_dict:  # Check for nested state dicts
            if reason := modules_have_similar_state_dict(
                sd1[key], sd2[key], rtol, atol
            ):
                return reason
        elif not torch.allclose(sd1[key], sd2[key], rtol=rtol, atol=atol):
            return f"Key {key} has different values: {sd1[key]} vs {sd2[key]}"

    return ""
