from dataclasses import dataclass
from typing import Optional

import torch

"""
Transition is designed for one single set of data
"""


@dataclass(frozen=False)
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: Optional[torch.Tensor] = None
    next_action: Optional[torch.Tensor] = None
    curr_available_actions: Optional[torch.Tensor] = None
    curr_available_actions_mask: Optional[torch.Tensor] = None
    next_available_actions: Optional[torch.Tensor] = None
    next_available_actions_mask: Optional[torch.Tensor] = None
    done: Optional[torch.Tensor] = None
    weight: Optional[torch.Tensor] = None


"""
TransitionBatch is designed for data batch
"""


@dataclass(frozen=False)
class TransitionBatch:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: Optional[torch.Tensor] = None
    next_action: Optional[torch.Tensor] = None
    curr_available_actions: Optional[torch.Tensor] = None
    curr_available_actions_mask: Optional[torch.Tensor] = None
    next_available_actions: Optional[torch.Tensor] = None
    next_available_actions_mask: Optional[torch.Tensor] = None
    done: Optional[torch.Tensor] = None
    weight: Optional[torch.Tensor] = None
