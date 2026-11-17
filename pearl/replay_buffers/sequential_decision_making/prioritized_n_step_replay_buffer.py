# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from collections import deque
from dataclasses import dataclass, fields

import torch
from pearl.api.action import Action
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch
from torch import Tensor


@dataclass(kw_only=True)
class PrioritizedNStepTransitionBatch(TransitionBatch):
    """A replay batch with importance weights and its actual bootstrap discounts.

    ``weight`` contains normalized importance-sampling weights. ``indices`` contains
    opaque transition IDs, not physical ring slots. Overwritten IDs are rejected.
    ``bootstrap_discount`` already includes both the actual horizon and termination.
    """

    indices: Tensor
    bootstrap_discount: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        batch_size = len(self)
        if self.indices.shape != (batch_size,) or self.indices.dtype != torch.long:
            raise ValueError("indices must be a one-dimensional int64 tensor")
        if self.bootstrap_discount.shape != (batch_size,):
            raise ValueError("bootstrap_discount must have one value per transition")
        if (
            not torch.isfinite(self.bootstrap_discount).all()
            or not (
                (self.bootstrap_discount >= 0) & (self.bootstrap_discount <= 1)
            ).all()
        ):
            raise ValueError("bootstrap discounts must be finite and between 0 and 1")


@dataclass
class _NStepRecord:
    transition: Transition
    bootstrap_discount: float


class PrioritizedNStepReplayBuffer(TensorBasedReplayBuffer):
    """Proportional prioritized replay with episode-aware n-step returns.

    Push one stream of discrete-action transitions in chronological order. A true
    termination or a truncation flushes all pending prefixes, so windows never cross
    episode boundaries. Truncations bootstrap from their supplied final observation;
    callers must not substitute an autoreset observation.

    Sampling is with replacement and costs O(capacity + batch_size), using an indexed
    CPU ring and a vector of priorities. This deliberately simple prototype does not
    implement a segment tree, concurrent collectors, replay checkpointing, or costs.
    Sampled transition IDs remain valid until their records are overwritten or cleared.
    This does not make the buffer safe for concurrent collectors.

    ``beta`` can be changed between updates to anneal the importance correction.
    Weights are normalized by the largest possible weight in the entire buffer.
    """

    def __init__(
        self,
        size: int,
        n_steps: int = 3,
        discount_factor: float = 0.99,
        alpha: float = 0.6,
        beta: float = 0.4,
        priority_epsilon: float = 1e-6,
    ) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("size must be a positive integer")
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")
        if not math.isfinite(discount_factor) or not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be finite and between 0 and 1")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and between 0 and 1")
        if not math.isfinite(priority_epsilon) or priority_epsilon <= 0:
            raise ValueError("priority_epsilon must be positive and finite")
        super().__init__(size)
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.beta = beta
        self.priority_epsilon = priority_epsilon
        self._pending: deque[Transition] = deque()
        self._storage: list[_NStepRecord] = []
        self._priorities = torch.zeros(size, dtype=torch.float64, device="cpu")
        self._next_index = 0
        self._transition_ids = torch.full((size,), -1, dtype=torch.long, device="cpu")
        self._next_transition_id = 0
        self._max_priority = 1.0

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("beta must be finite and between 0 and 1")
        self._beta = value

    @staticmethod
    def _copy_to_cpu(tensor: Tensor) -> Tensor:
        return tensor.detach().to(device="cpu").clone()

    def _store_transition(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        truncated: bool,
        curr_available_actions_tensor_with_padding: Tensor | None,
        curr_unavailable_actions_mask: Tensor | None,
        next_state: SubjectiveState | None,
        next_available_actions_tensor_with_padding: Tensor | None,
        next_unavailable_actions_mask: Tensor | None,
        cost: float | None = None,
    ) -> None:
        if self.is_action_continuous:
            raise ValueError("PrioritizedNStepReplayBuffer requires discrete actions")
        if cost is not None:
            raise ValueError(
                "This n-step replay prototype does not support cost metadata"
            )
        if (
            curr_available_actions_tensor_with_padding is None
            or curr_unavailable_actions_mask is None
        ):
            raise ValueError("Current available actions must be provided")
        if not terminated and (
            next_state is None
            or next_available_actions_tensor_with_padding is None
            or next_unavailable_actions_mask is None
        ):
            raise ValueError(
                "Bootstrapping requires a next state and its available actions"
            )

        state_tensor = torch.as_tensor(state).detach().cpu().clone().unsqueeze(0)
        next_state_tensor = (
            torch.zeros_like(state_tensor)
            if next_state is None
            else torch.as_tensor(next_state).detach().cpu().clone().unsqueeze(0)
        )
        # Terminal placeholders make mixed batches well-formed. Their discount is zero.
        if next_available_actions_tensor_with_padding is None:
            next_available_actions_tensor_with_padding = (
                curr_available_actions_tensor_with_padding
            )
        if next_unavailable_actions_mask is None:
            next_unavailable_actions_mask = curr_unavailable_actions_mask
        transition = Transition(
            state=state_tensor,
            action=torch.as_tensor(action).detach().cpu().clone().unsqueeze(0),
            reward=torch.tensor([reward], dtype=torch.float32, device="cpu"),
            terminated=torch.tensor([terminated], dtype=torch.bool, device="cpu"),
            truncated=torch.tensor([truncated], dtype=torch.bool, device="cpu"),
            next_state=next_state_tensor,
            curr_available_actions=self._copy_to_cpu(
                curr_available_actions_tensor_with_padding
            ),
            curr_unavailable_actions_mask=self._copy_to_cpu(
                curr_unavailable_actions_mask
            ),
            next_available_actions=self._copy_to_cpu(
                next_available_actions_tensor_with_padding
            ),
            next_unavailable_actions_mask=self._copy_to_cpu(
                next_unavailable_actions_mask
            ),
        )
        self._pending.append(transition)
        if len(self._pending) == self.n_steps:
            self._emit_oldest_prefix()
        if terminated or truncated:
            while self._pending:
                self._emit_oldest_prefix()

    def _emit_oldest_prefix(self) -> None:
        if self._next_transition_id >= torch.iinfo(torch.long).max:
            raise OverflowError("replay transition IDs exhausted")
        first = self._pending[0]
        last = self._pending[-1]
        reward = torch.zeros_like(first.reward)
        for step, transition in enumerate(self._pending):
            reward += (self.discount_factor**step) * transition.reward
        discount = (
            0.0
            if bool(last.terminated.item())
            else self.discount_factor ** len(self._pending)
        )
        record = _NStepRecord(
            transition=Transition(
                state=first.state,
                action=first.action,
                reward=reward,
                terminated=last.terminated,
                truncated=last.truncated,
                next_state=last.next_state,
                curr_available_actions=first.curr_available_actions,
                curr_unavailable_actions_mask=first.curr_unavailable_actions_mask,
                next_available_actions=last.next_available_actions,
                next_unavailable_actions_mask=last.next_unavailable_actions_mask,
            ),
            bootstrap_discount=discount,
        )
        if len(self._storage) < self.capacity:
            self._storage.append(record)
        else:
            self._storage[self._next_index] = record
        self._priorities[self._next_index] = self._max_priority
        self._transition_ids[self._next_index] = self._next_transition_id
        self._next_transition_id += 1
        self._next_index = (self._next_index + 1) % self.capacity
        self._pending.popleft()

    def _sampling_probabilities(self) -> Tensor:
        priorities = self._priorities[: len(self)]
        # Rescaling before the power avoids overflowing a sum of large priorities.
        scores = (priorities / priorities.max()).clamp_min(
            torch.finfo(torch.float64).tiny
        ) ** self.alpha
        return scores / scores.sum()

    def sample(self, batch_size: int) -> PrioritizedNStepTransitionBatch:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not self._storage:
            raise ValueError("Cannot sample from an empty replay buffer")
        probabilities = self._sampling_probabilities()
        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        records = [self._storage[index] for index in indices.tolist()]
        batch = self._create_transition_batch(
            [record.transition for record in records], is_action_continuous=False
        )
        # (N P(i))**(-beta) / max_j (N P(j))**(-beta), evaluated in log space.
        weights = torch.exp(
            -self.beta * (probabilities[indices].log() - probabilities.min().log())
        ).to(dtype=torch.float32, device=self.device_for_batches)
        batch_fields = {
            field.name: getattr(batch, field.name) for field in fields(TransitionBatch)
        }
        batch_fields["weight"] = weights
        return PrioritizedNStepTransitionBatch(
            **batch_fields,
            indices=self._transition_ids[indices].to(device=self.device_for_batches),
            bootstrap_discount=torch.tensor(
                [record.bootstrap_discount for record in records],
                dtype=torch.float32,
                device=self.device_for_batches,
            ),
        )

    def update_priorities(self, indices: Tensor, priorities: Tensor) -> None:
        """Update sampled transition IDs from unweighted per-transition losses.

        Duplicate samples use the largest supplied priority, independently of order.
        All inputs are checked before any priorities are changed.
        """
        indices = torch.as_tensor(indices).detach().cpu()
        priorities = (
            torch.as_tensor(priorities).detach().to(device="cpu", dtype=torch.float64)
        )
        if (
            indices.ndim != 1
            or priorities.ndim != 1
            or indices.shape != priorities.shape
        ):
            raise ValueError(
                "indices and priorities must be equally sized one-dimensional tensors"
            )
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must contain integers")
        slots = indices.remainder(self.capacity)
        if (indices < 0).any() or (self._transition_ids[slots] != indices).any():
            raise ValueError("A replay transition ID is stale or out of range")
        if not torch.isfinite(priorities).all() or (priorities < 0).any():
            raise ValueError("priorities must be finite and nonnegative")
        updates: dict[int, float] = {}
        for index, value in zip(slots.tolist(), priorities.tolist()):
            adjusted_priority = value + self.priority_epsilon
            if not math.isfinite(adjusted_priority):
                raise ValueError("priority plus priority_epsilon must remain finite")
            updates[index] = max(updates.get(index, 0.0), adjusted_priority)
        for index, value in updates.items():
            self._priorities[index] = value
            self._max_priority = max(self._max_priority, value)

    def __len__(self) -> int:
        return len(self._storage)

    def clear(self) -> None:
        self._storage.clear()
        self._pending.clear()
        self._priorities.zero_()
        self._next_index = 0
        self._transition_ids.fill_(-1)
        # IDs must never alias records that existed before clear(). Align the
        # next ID to slot zero while preserving monotonically increasing IDs.
        self._next_transition_id = (
            (self._next_transition_id + self.capacity - 1) // self.capacity
        ) * self.capacity
        self._max_priority = 1.0
