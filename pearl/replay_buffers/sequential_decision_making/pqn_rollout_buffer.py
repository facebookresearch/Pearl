# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""One fresh chronological PQN rollout, not experience replay."""

from dataclasses import dataclass, fields
from typing import Any

import torch
from pearl.api.action_space import ActionSpace
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


@dataclass
class PQNRollout:
    """Time-major data; next_states are final observations, never reset states.

    Terminal placeholders may be nonfinite because the learner never queries them.
    A truncation bootstraps from its final observation but stops the lambda trace.
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_states: torch.Tensor
    next_unavailable_actions_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            self.states.ndim != 3
            or self.states.shape[0] < 1
            or self.states.shape[1] < 1
        ):
            raise ValueError(
                "states must have nonempty [time, environments, features] axes"
            )
        shape = self.states.shape[:2]
        if (
            self.actions.shape != (*shape, 1)
            or self.next_states.shape != self.states.shape
        ):
            raise ValueError("action and next-state axes must match the rollout")
        if (
            self.actions.dtype == torch.bool
            or self.actions.is_complex()
            or not torch.isfinite(self.actions).all()
            or not torch.equal(self.actions, self.actions.round())
            or (self.actions < 0).any()
        ):
            raise ValueError("actions must be finite nonnegative integer catalog IDs")
        for name in ("rewards", "terminated", "truncated"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape [time, environments]")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise ValueError("termination and truncation flags must be boolean")
        if (
            not torch.isfinite(self.states).all()
            or not torch.isfinite(self.rewards).all()
        ):
            raise ValueError("current states and rewards must be finite")
        if not torch.isfinite(self.next_states[~self.terminated]).all():
            raise ValueError("bootstrapped final next states must be finite")
        mask = self.next_unavailable_actions_mask
        if mask is not None and (
            mask.ndim != 3 or mask.shape[:2] != shape or mask.dtype != torch.bool
        ):
            raise ValueError(
                "next-action mask must be boolean [time, environments, catalog]"
            )

    def to(self, device: torch.device) -> "PQNRollout":
        return PQNRollout(
            **{
                f.name: (
                    getattr(self, f.name).to(device)
                    if getattr(self, f.name) is not None
                    else None
                )
                for f in fields(self)
            }
        )

    def __len__(self) -> int:
        return self.states.shape[0] * self.states.shape[1]


@dataclass(kw_only=True)
class PQNTransitionBatch(TransitionBatch):
    """Flattened observations/actions and detached chronological Q(lambda) targets."""

    returns: torch.Tensor


class PQNRolloutBuffer(ReplayBuffer):
    """Compatibility adapter for one serial PearlAgent stream.

    The learner consumes and clears this buffer after each learning call. There
    is no sampling of older experience. Vector collectors construct PQNRollout
    directly and need no replay object. A failed learning call leaves data intact.
    """

    def __init__(self, max_steps: int = 128) -> None:
        super().__init__()
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self._device_for_batches = torch.device("cpu")
        self._rows: list[dict[str, Any]] = []

    @property
    def device_for_batches(self) -> torch.device:
        return self._device_for_batches

    @device_for_batches.setter
    def device_for_batches(self, value: torch.device) -> None:
        self._device_for_batches = torch.device(value)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
        curr_available_actions: ActionSpace | None = None,
        next_state: torch.Tensor | None = None,
        next_available_actions: ActionSpace | None = None,
        max_number_actions: int | None = None,
        cost: float | None = None,
    ) -> None:
        if len(self._rows) >= self.max_steps:
            raise ValueError(
                "rollout is full; consume it before collecting more transitions"
            )
        if self.is_action_continuous or cost is not None:
            raise ValueError("PQN rollout supports reward-only discrete actions")
        state = torch.as_tensor(state).detach().cpu().reshape(-1).clone()
        if next_state is None and not terminated:
            raise ValueError(
                "bootstrapping needs the transition's final next observation"
            )
        if (
            not isinstance(next_available_actions, DiscreteActionSpace)
            or next_available_actions.action_dim != 1
            or isinstance(max_number_actions, bool)
            or not isinstance(max_number_actions, int)
            or max_number_actions < 1
        ):
            raise ValueError("fixed discrete next-action catalog metadata is required")
        ids = next_available_actions.actions_batch.reshape(-1)
        if (
            ids.dtype == torch.bool
            or ids.is_complex()
            or not torch.isfinite(ids).all()
            or not torch.equal(ids, ids.round())
            or (ids < 0).any()
            or (ids >= max_number_actions).any()
            or ids.unique().numel() != ids.numel()
        ):
            raise ValueError("next actions must be unique IDs within the fixed catalog")
        mask = torch.ones(max_number_actions, dtype=torch.bool)
        mask[ids.long().cpu()] = False
        self._rows.append(
            dict(
                states=state,
                actions=torch.as_tensor(action).detach().cpu().reshape(1).clone(),
                rewards=torch.tensor(float(reward)),
                terminated=torch.tensor(bool(terminated)),
                truncated=torch.tensor(bool(truncated)),
                next_states=(
                    torch.zeros_like(state)
                    if next_state is None
                    else torch.as_tensor(next_state).detach().cpu().reshape(-1).clone()
                ),
                next_unavailable_actions_mask=mask,
            )
        )

    def as_rollout(self) -> PQNRollout:
        if not self._rows:
            raise ValueError("empty rollout")
        return PQNRollout(
            **{
                name: torch.stack([r[name] for r in self._rows]).unsqueeze(1)
                for name in self._rows[0]
            }
        ).to(self.device_for_batches)

    def sample(self, batch_size: int) -> object:
        raise TypeError(
            "PQN consumes a chronological rollout; random replay samples are invalid"
        )

    def clear(self) -> None:
        self._rows.clear()

    def __len__(self) -> int:
        return len(self._rows)
