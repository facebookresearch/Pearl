# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import dataclasses
from dataclasses import dataclass
from typing import Optional, TypeVar

import torch
from torch import Tensor


T = TypeVar("T", bound="Transition")


@dataclass(frozen=False)
class Transition:
    """
    Transition is designed for one single set of data
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    next_state: Optional[torch.Tensor] = None
    next_action: Optional[torch.Tensor] = None
    curr_available_actions: Optional[torch.Tensor] = None
    curr_unavailable_actions_mask: Optional[torch.Tensor] = None
    next_available_actions: Optional[torch.Tensor] = None
    next_unavailable_actions_mask: Optional[torch.Tensor] = None
    weight: Optional[torch.Tensor] = None
    cum_reward: Optional[torch.Tensor] = None
    cost: Optional[torch.Tensor] = None

    def to(self: T, device: torch.device) -> T:
        # iterate over all fields, move to correct device
        for f in dataclasses.fields(self.__class__):
            if getattr(self, f.name) is not None:
                super().__setattr__(
                    f.name,
                    torch.as_tensor(getattr(self, f.name)).to(device),
                )
        return self

    @property
    def device(self) -> torch.device:
        return self.state.device


TB = TypeVar("TB", bound="TransitionBatch")


@dataclass(frozen=False)
class TransitionBatch:
    """
    TransitionBatch is designed for data batch
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    next_state: Optional[torch.Tensor] = None
    next_action: Optional[torch.Tensor] = None
    curr_available_actions: Optional[torch.Tensor] = None
    curr_unavailable_actions_mask: Optional[torch.Tensor] = None
    next_available_actions: Optional[torch.Tensor] = None
    next_unavailable_actions_mask: Optional[torch.Tensor] = None
    weight: Optional[torch.Tensor] = None
    cum_reward: Optional[torch.Tensor] = None
    time_diff: Optional[torch.Tensor] = None
    cost: Optional[torch.Tensor] = None

    def to(self: TB, device: torch.device) -> TB:
        # iterate over all fields
        for f in dataclasses.fields(self.__class__):
            if getattr(self, f.name) is not None:
                item = getattr(self, f.name)
                item = torch.as_tensor(item, device=device)
                super().__setattr__(
                    f.name,
                    item,
                )
        return self

    @property
    def device(self) -> torch.device:
        """
        The device where the batch lives.
        """
        return self.state.device

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
    mask: Optional[torch.Tensor] = batch.bootstrap_mask

    def _filter_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or mask is None:
            return None
        return x[mask[:, z] == 1]

    filtered_state = _filter_tensor(batch.state)
    filtered_action = _filter_tensor(batch.action)
    filtered_reward = _filter_tensor(batch.reward)
    filtered_done = _filter_tensor(batch.done)

    assert filtered_state is not None
    assert filtered_action is not None
    assert filtered_reward is not None
    assert filtered_done is not None

    return TransitionBatch(
        state=filtered_state,
        action=filtered_action,
        reward=filtered_reward,
        done=filtered_done,
        next_state=_filter_tensor(batch.next_state),
        next_action=_filter_tensor(batch.next_action),
        curr_available_actions=_filter_tensor(batch.curr_available_actions),
        curr_unavailable_actions_mask=_filter_tensor(
            batch.curr_unavailable_actions_mask
        ),
        next_available_actions=_filter_tensor(batch.next_available_actions),
        next_unavailable_actions_mask=_filter_tensor(
            batch.next_unavailable_actions_mask
        ),
        weight=_filter_tensor(batch.weight),
        cum_reward=_filter_tensor(batch.cum_reward),
        cost=_filter_tensor(batch.cost),
    ).to(batch.device)
