import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch

from pearl.utils.device import get_pearl_device


@dataclass(frozen=False)
class Transition:
    """
    Transition is designed for one single set of data
    """

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

    # pyre-fixme[3]: Return type must be annotated.
    def __post_init__(self):
        pearl_device = get_pearl_device()
        # iterate over all fields, move to correct device
        for field in dataclasses.fields(Transition):
            if getattr(self, field.name) is not None:
                super().__setattr__(
                    field.name, getattr(self, field.name).to(pearl_device)
                )


@dataclass(frozen=False)
class TransitionBatch:
    """
    TransitionBatch is designed for data batch
    """

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

    # pyre-fixme[3]: Return type must be annotated.
    def __post_init__(self):
        pearl_device = get_pearl_device()
        # iterate over all fields, move to correct device
        for field in dataclasses.fields(TransitionBatch):
            if getattr(self, field.name) is not None:
                super().__setattr__(
                    field.name, getattr(self, field.name).to(pearl_device)
                )
