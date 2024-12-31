# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import dataclasses
from dataclasses import dataclass
from typing import TypeVar

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
    terminated: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    truncated: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    next_state: torch.Tensor | None = None
    next_action: torch.Tensor | None = None
    curr_available_actions: torch.Tensor | None = None
    curr_unavailable_actions_mask: torch.Tensor | None = None
    next_available_actions: torch.Tensor | None = None
    next_unavailable_actions_mask: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    cost: torch.Tensor | None = None

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
    terminated: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    truncated: torch.Tensor = torch.tensor(True)  # default True is useful for bandits
    next_state: torch.Tensor | None = None
    next_action: torch.Tensor | None = None
    curr_available_actions: torch.Tensor | None = None
    curr_unavailable_actions_mask: torch.Tensor | None = None
    next_available_actions: torch.Tensor | None = None
    next_unavailable_actions_mask: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    time_diff: torch.Tensor | None = None
    cost: torch.Tensor | None = None

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
    bootstrap_mask: torch.Tensor | None = None


@dataclass(frozen=False)
class TransitionWithBootstrapMaskBatch(TransitionBatch):
    bootstrap_mask: torch.Tensor | None = None


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
    mask: torch.Tensor | None = batch.bootstrap_mask

    def _filter_tensor(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None or mask is None:
            return None
        return x[mask[:, z] == 1]

    filtered_state = _filter_tensor(batch.state)
    filtered_action = _filter_tensor(batch.action)
    filtered_reward = _filter_tensor(batch.reward)
    filtered_terminated = _filter_tensor(batch.terminated)
    filtered_truncated = _filter_tensor(batch.truncated)

    assert filtered_state is not None
    assert filtered_action is not None
    assert filtered_reward is not None
    assert filtered_terminated is not None
    assert filtered_truncated is not None

    return TransitionBatch(
        state=filtered_state,
        action=filtered_action,
        reward=filtered_reward,
        terminated=filtered_terminated,
        truncated=filtered_truncated,
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
        cost=_filter_tensor(batch.cost),
    ).to(batch.device)
