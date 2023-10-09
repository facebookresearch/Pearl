import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

import torch

from pearl.utils.device import get_pearl_device
from torch import Tensor


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
        for field in dataclasses.fields(self.__class__):
            if getattr(self, field.name) is not None:
                super().__setattr__(
                    field.name,
                    torch.as_tensor(getattr(self, field.name)).to(pearl_device),
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
        # iterate over all fields
        for field in dataclasses.fields(self.__class__):
            if getattr(self, field.name) is not None:
                item = getattr(self, field.name)
                if (
                    isinstance(item, np.ndarray)
                    or isinstance(item, float)
                    or isinstance(item, int)
                ):
                    item = torch.tensor(item)  # convert to tensor if it wasn't a tensor
                item = item.to(pearl_device)  # move to correct device
                super().__setattr__(
                    field.name,
                    item,
                )

    def __len__(self) -> int:
        return self.reward.shape[0]


@dataclass(frozen=False)
class TransitionWithBootstrapMask(Transition):
    bootstrap_mask: Optional[torch.Tensor] = None


@dataclass(frozen=False)
class TransitionWithBootstrapMaskBatch(TransitionBatch):
    bootstrap_mask: Optional[torch.Tensor] = None


def filter_batch_by_bootstrap_mask(
    batch: TransitionWithBootstrapMaskBatch, z: Tensor
) -> TransitionBatch:
    r"""A helper function that filters a `TransitionBatch` to only those transitions
    that are marked as active (by its `bootstrap_mask` field) for a given ensemble
    index `z`.

    Args:
        batch: The original `TransitionWithBootstrapMask`.
        z: The ensemble index to filter on.

    Returns:
        A filtered `TransitionBatch`.
    """
    mask = batch.bootstrap_mask

    # pyre-ignore[53]
    def _filter_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or mask is None:
            return None
        return x[mask[:, z] == 1]

    return TransitionBatch(
        state=_filter_tensor(batch.state),  # pyre-ignore
        action=_filter_tensor(batch.action),  # pyre-ignore
        reward=_filter_tensor(batch.reward),  # pyre-ignore
        next_state=_filter_tensor(batch.next_state),
        next_action=_filter_tensor(batch.next_action),
        curr_available_actions=_filter_tensor(batch.curr_available_actions),
        curr_available_actions_mask=_filter_tensor(batch.curr_available_actions_mask),
        next_available_actions=_filter_tensor(batch.next_available_actions),
        next_available_actions_mask=_filter_tensor(batch.next_available_actions_mask),
        done=_filter_tensor(batch.done),
        weight=_filter_tensor(batch.weight),
    )
