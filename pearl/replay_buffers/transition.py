# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import dataclasses
from dataclasses import dataclass, field
from typing import cast, Final, TypeVar

import torch
from torch import Tensor
from torch.fx._symbolic_trace import is_fx_tracing


T = TypeVar("T", bound="Transition")


@dataclass(frozen=False)
class Transition:
    """
    Transition is designed for one single set of data

    Args:
        state (torch.Tensor): Tensor of shape (state_dim) representing the current state.
        action (torch.Tensor): Tensor of shape (action_dim) representing the action taken.
        reward (torch.Tensor): Tensor of shape (1) representing the reward received.
        terminated (boolean torch.Tensor): Tensor of shape (1), default True is useful for bandits.
          which can be seen as a sequence being immediately terminated.
        truncated (boolean torch.Tensor): Tensor of shape (1), default False is useful for bandits.
          which can be seen as a sequence being immediately terminated (and thus not truncated).
        next_state (torch.Tensor | None): Tensor of shape (state_dim).
        next_action (torch.Tensor | None): Tensor of shape (action_dim).
        curr_available_actions (torch.Tensor | None): shape (action_space_size x action_dim)
        curr_unavailable_actions_mask (torch.Tensor | None): Tensor of shape (action_space_size)
        next_available_actions (torch.Tensor | None): shape (action_space_size x action_dim)
            representing next available actions.
        next_unavailable_actions_mask (torch.Tensor | None): Tensor of shape (action_space_size)
            representing mask for next unavailable actions.
        weight (torch.Tensor | None): Tensor of shape (1) representing the weight of the transition.
        cost (torch.Tensor | None): Tensor of shape (1); the cost associated with the transition.
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor = torch.tensor(True)
    truncated: torch.Tensor = torch.tensor(False)
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

# In dataclasses, the __init__ arguments and the corresponding attributes
# have the same type. If we make the type of the argument Tensor | None,
# the attribute will also have that type, even though at runtime they
# will never be None (because they receive default values).
# So this would require users to constantly write batch.terminated is
# not None before use to avoid Pyre complaints.
# The way this is written now, we use _UNSET as a sentinel cast to
# a Tensor, so both the argument and the attribute are type Tensor,
# making usage simpler.
_UNSET: Final = object()


@dataclass(frozen=False)
class TransitionBatch:
    """
    TransitionBatch is designed for data batch

    Args:
        state (torch.Tensor): Tensor of shape (batch_size x state_dim)
        action (torch.Tensor): Tensor of shape (batch_size x action_dim)
        reward (torch.Tensor): Tensor of shape (batch_size)
        terminated (torch.Tensor): Tensor, default True is useful for bandits
          which can be seen as a sequence being immediately terminated.
        truncated (torch.Tensor): Tensor, default False is useful for bandits
          which can be seen as a sequence being immediately terminated (not truncated)
        next_state (torch.Tensor | None): Tensor of shape (batch_size x state_dim)
        next_action (torch.Tensor | None): Tensor
        curr_available_actions (torch.Tensor | None): Tensor of shape
            (batch_size x action_space_size x action_dim)
        curr_unavailable_actions_mask (torch.Tensor | None): Tensor
        next_available_actions (torch.Tensor | None): Tensor of shape
            (batch_size x action_space_size x action_dim)
        next_unavailable_actions_mask (torch.Tensor | None): Tensor of shape
            (batch_size x action_space_size)
        weight (torch.Tensor | None): Tensor
        time_diff (torch.Tensor | None): Tensor
        cost (torch.Tensor | None): Tensor
    """

    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    # See comment for _UNSET above
    terminated: torch.Tensor = field(default=cast(torch.Tensor, _UNSET))
    truncated: torch.Tensor = field(default=cast(torch.Tensor, _UNSET))
    next_state: torch.Tensor | None = None
    next_action: torch.Tensor | None = None
    curr_available_actions: torch.Tensor | None = None
    curr_unavailable_actions_mask: torch.Tensor | None = None
    next_available_actions: torch.Tensor | None = None
    next_unavailable_actions_mask: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    time_diff: torch.Tensor | None = None
    cost: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """
        Post-initialization validation checks.
        It also sets the default values for terminated and truncated if needed.

        This method performs the following checks:

        - The state has at least 2 dimensions (batch_size, ...).
        - The action has shape (batch_size,) or (batch_size, ...).
        - The reward has shape (batch_size,) or (batch_size, 1).
        - The state and reward have the same batch_size dimension.
        - The terminated and truncated tensors have shape (batch_size,) or (batch_size, 1).
        - The next_state and next_action tensors have at least 2 dimensions (batch_size, ...).
        """

        # Bypass meaningless post init during fx tracing as symbolically traced variables
        # cannot be used as inputs to control flow
        if is_fx_tracing():
            return

        assert self.state.ndim >= 2, (
            f"state has shape {self.state.shape}, "
            f"but must have at least 2 dimensions (batch_size, ...)"
        )

        # Allow action to have shape (batch_size,) or (batch_size, ...)
        assert (
            self.action.ndim >= 1
        ), f"action has shape {self.action.shape}, but must be (batch_size,) or (batch_size, ...)"

        # Allow reward to have shape (batch_size,) or (batch_size, ...)
        assert (
            self.reward.ndim >= 1
        ), f"reward has shape {self.reward.shape}, but must be (batch_size,) or (batch_size, ...)"

        assert self.state.shape[0] == self.reward.shape[0], (
            f"state has shape {self.state.shape}, "
            f"but reward has shape {self.reward.shape}, "
            f"and they must have the same batch_size dimension"
        )

        batch_size = self.reward.shape[0]

        if self.terminated is not _UNSET:
            assert (
                self.terminated.ndim == 1 and self.terminated.shape[0] == batch_size
            ) or (
                self.terminated.ndim == 2 and self.terminated.shape == (batch_size, 1)
            ), (
                f"terminated has shape {self.terminated.shape} but it should be equal to "
                f"either ({batch_size},) or ({batch_size}, 1) (since batch_size is {batch_size})"
            )
        else:
            # Always create terminated with shape (batch_size,) regardless of reward shape
            self.terminated = torch.ones(
                batch_size, dtype=torch.bool, device=self.reward.device
            )

        if self.truncated is not _UNSET:
            assert (
                self.truncated.ndim == 1 and self.truncated.shape[0] == batch_size
            ) or (
                self.truncated.ndim == 2 and self.truncated.shape == (batch_size, 1)
            ), (
                f"truncated has shape {self.truncated.shape} but it should be equal to "
                f"either ({batch_size},) or ({batch_size}, 1) (since batch_size is {batch_size})"
            )
        else:
            # Always create truncated with shape (batch_size,) regardless of reward shape
            self.truncated = torch.zeros(
                batch_size, dtype=torch.bool, device=self.reward.device
            )

        if self.next_state is not None:
            assert self.next_state.ndim >= 2, (
                f"next_state has shape {self.next_state.shape}, "
                f"but must have at least 2 dimensions ({batch_size}, ...) "
                f"(since batch_size is {batch_size})"
            )
        if self.next_action is not None:
            # Allow next_action to have shape (batch_size,) or (batch_size, ...)
            assert self.next_action.ndim >= 1, (
                f"next_action has shape {self.next_action.shape}, "
                f"but must be ({batch_size},) or ({batch_size}, ...) "
                f"(since batch_size is {batch_size})"
            )

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
