from dataclasses import dataclass
from typing import Optional

import torch

"""
Transition is designed for one single set of data
"""


@dataclass(frozen=True)
class Transition:
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor
    next_action: Optional[torch.tensor] = None
    curr_available_actions: torch.tensor = None
    curr_available_actions_mask: torch.tensor = None
    next_available_actions: torch.tensor = None
    next_available_actions_mask: torch.tensor = None
    done: torch.tensor = None
    weight: torch.tensor = None


"""
TransitionBatch is designed for data batch
"""


@dataclass(frozen=True)
class TransitionBatch:
    state: torch.tensor
    action: torch.tensor
    reward: torch.tensor
    next_state: torch.tensor = None
    next_action: Optional[torch.tensor] = None
    curr_available_actions: torch.tensor = None
    curr_available_actions_mask: torch.tensor = None
    next_available_actions: torch.tensor = None
    next_available_actions_mask: torch.tensor = None
    done: torch.tensor = None
    weight: torch.tensor = None
