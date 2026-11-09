# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Parallel Q-learning (Gallici et al., arXiv:2407.04811).

The vector setup follows CleanRL fe8d8a03c41a7ef5b523e2e354bd01c363e786bb.
Time limits bootstrap from final observations and stop the return trace,
rather than treating every episode boundary as a true terminal.
"""

import copy
import math
from typing import Any

import torch
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.history_summarization_modules.identity_history_summarization_module import (
    IdentityHistorySummarizationModule,
)
from pearl.neural_networks.sequential_decision_making.parallel_q_value_network import (
    ParallelQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.pqn_rollout_buffer import (
    PQNRollout,
    PQNRolloutBuffer,
    PQNTransitionBatch,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.identity_safety_module import IdentitySafetyModule
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


@torch.no_grad()
def q_lambda_returns(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    discount_factor: float,
    trace_decay: float,
) -> torch.Tensor:
    """Finite-horizon off-policy lambda returns, time-major [time, environments].

    At truncations and the end of a rollout the target is a one-step bootstrap.
    True terminals have no bootstrap. No return ever crosses an episode boundary.
    Caller supplies the next state's masked greedy value from the unchanged online
    network, using a final observation rather than an autoreset observation.
    """
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("rewards must have nonempty [time, environments] axes")
    if any(
        value.shape != rewards.shape for value in [next_values, terminated, truncated]
    ):
        raise ValueError("return inputs must have identical shapes")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("episode flags must be boolean")
    if rewards.is_complex() or next_values.is_complex():
        raise ValueError("rewards and next values must be real")
    dtype = torch.promote_types(rewards.dtype, next_values.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    rewards = rewards.to(dtype=dtype)
    next_values = next_values.to(dtype=dtype)
    if not (
        math.isfinite(discount_factor)
        and 0 <= discount_factor <= 1
        and math.isfinite(trace_decay)
        and 0 <= trace_decay <= 1
    ):
        raise ValueError("discount and trace decay must be finite and in [0, 1]")
    if (
        not torch.isfinite(rewards).all()
        or not torch.isfinite(next_values[~terminated]).all()
    ):
        raise ValueError("rewards and bootstrapped next values must be finite")
    safe_values = next_values.masked_fill(terminated, 0)
    result = torch.empty_like(rewards)
    following_return = safe_values[-1]
    for index in range(rewards.shape[0] - 1, -1, -1):
        continue_trace = ~(terminated[index] | truncated[index])
        mixed = (1 - trace_decay) * safe_values[index] + trace_decay * following_return
        bootstrap = torch.where(continue_trace, mixed, safe_values[index])
        result[index] = rewards[index] + discount_factor * bootstrap
        following_return = result[index]
    return result


class ParallelQLearning(PolicyLearner):
    """PQN with LayerNorm, RAdam and fresh vector rollouts.

    Each update computes fixed Q(lambda) targets, then fits the current
    rollout for the configured number of shuffled epochs. The online network
    supplies targets; there is no frozen target network or persistent replay.
    on_policy=False describes the greedy off-policy target. learn consumes
    the temporary rollout buffer.

    act supports one PearlAgent stream. act_batch and learn_rollout support
    batched inference and learning across independent environments.
    Requires fixed scalar action IDs and identity history, action and safety
    modules. Checkpoints do not capture the environment, RNG or live rollout.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: DiscreteActionSpace,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 2.5e-4,
        discount_factor: float = 0.99,
        trace_decay: float = 0.65,
        update_epochs: int = 4,
        batch_size: int = 128,
        epsilon: float = 0.05,
        max_grad_norm: float | None = 10.0,
        weight_decay: float = 0.0,
        network_instance: QValueNetwork | None = None,
    ) -> None:
        if not isinstance(action_space, DiscreteActionSpace):
            raise TypeError("PQN requires a fixed DiscreteActionSpace")
        ids = action_space.actions_batch
        if (
            action_space.action_dim != 1
            or ids.dtype == torch.bool
            or ids.is_complex()
            or not torch.equal(
                ids.detach().cpu().reshape(-1), torch.arange(action_space.n)
            )
        ):
            raise ValueError("action space must be ordered scalar catalog 0, ..., n-1")
        for name, value in [
            ("update_epochs", update_epochs),
            ("batch_size", batch_size),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in [
            ("discount_factor", discount_factor),
            ("trace_decay", trace_decay),
            ("epsilon", epsilon),
        ]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not math.isfinite(learning_rate)
            or learning_rate < 0
            or not math.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise ValueError(
                "learning rate and weight decay must be finite and nonnegative"
            )
        if max_grad_norm is not None and (
            not math.isfinite(max_grad_norm) or max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")
        hidden_dims = [120, 84] if hidden_dims is None else list(hidden_dims)
        super().__init__(
            on_policy=False,
            is_action_continuous=False,
            action_space=action_space,
            training_rounds=update_epochs,
            batch_size=batch_size,
        )
        self._num_actions = action_space.n
        self._Q = (
            network_instance
            if network_instance is not None
            else ParallelQValueNetwork(state_dim, action_space.n, hidden_dims)
        )
        if self._Q.state_dim != state_dim or self._Q.action_dim != 1:
            raise ValueError(
                "Q network dimensions must match state and scalar action IDs"
            )
        self._optimizer = torch.optim.RAdam(
            self._Q.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self._discount_factor, self._trace_decay = discount_factor, trace_decay
        self._epsilon, self._max_grad_norm = epsilon, max_grad_norm
        self._environment_steps = 0
        self.register_buffer(
            "_catalog_actions", torch.arange(action_space.n).reshape(-1, 1)
        )
        self.safety_module = IdentitySafetyModule()
        self._checkpoint_config = dict(
            state_dim=state_dim,
            num_actions=action_space.n,
            hidden_dims=hidden_dims,
            discount_factor=discount_factor,
            trace_decay=trace_decay,
            update_epochs=update_epochs,
            batch_size=batch_size,
            epsilon=epsilon,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
        )

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        if type(value) is not IdentityHistorySummarizationModule:
            raise TypeError("PQN currently requires independent identity histories")
        self._history_summarization_module = value

    def _check_components(self) -> None:
        representation = self.action_representation_module
        if (
            type(representation) is not IdentityActionRepresentationModule
            or representation.representation_dim != 1
            or representation.max_number_actions != self._num_actions
            or any(p.requires_grad for p in representation.parameters())
        ):
            raise TypeError("PQN requires fixed identity action representation")
        if type(self.safety_module) is not IdentitySafetyModule:
            raise TypeError("PQN currently supports IdentitySafetyModule only")
        if (
            type(self._history_summarization_module)
            is not IdentityHistorySummarizationModule
        ):
            raise TypeError("PQN currently requires identity histories")

    def _space_ids(self, action_space: ActionSpace) -> torch.Tensor:
        if (
            not isinstance(action_space, DiscreteActionSpace)
            or action_space.action_dim != 1
        ):
            raise ValueError("available actions must be scalar catalog IDs")
        ids = action_space.actions_batch.reshape(-1)
        if (
            ids.numel() == 0
            or ids.dtype == torch.bool
            or ids.is_complex()
            or not torch.isfinite(ids).all()
            or not torch.equal(ids, ids.round())
            or (ids < 0).any()
            or (ids >= self._num_actions).any()
            or ids.unique().numel() != ids.numel()
        ):
            raise ValueError("available actions must be unique IDs within the catalog")
        return ids.to(device=self._catalog_actions.device, dtype=torch.long)

    def reset(self, action_space: ActionSpace) -> None:
        self._space_ids(action_space)

    def _all_values(self, states: torch.Tensor) -> torch.Tensor:
        actions = self._catalog_actions.unsqueeze(0).expand(states.shape[0], -1, -1)
        return self._Q.get_q_values(states, actions)

    @torch.no_grad()
    def act_batch(
        self,
        states: torch.Tensor,
        unavailable_actions_mask: torch.Tensor | None = None,
        exploit: bool = False,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        self._check_components()
        value = self._epsilon if epsilon is None else epsilon
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("epsilon must be finite and in [0, 1]")
        values = self._all_values(states)
        mask = (
            torch.zeros_like(values, dtype=torch.bool)
            if unavailable_actions_mask is None
            else unavailable_actions_mask
        )
        if mask.shape != values.shape or mask.dtype != torch.bool or mask.all(-1).any():
            raise ValueError(
                "each state needs a boolean catalog mask with an available action"
            )
        if not torch.isfinite(values[~mask]).all():
            raise ValueError("nonfinite Q value for an available action")
        actions = values.masked_fill(mask, -torch.inf).argmax(-1)
        if not exploit:
            self._environment_steps += states.shape[0]
            if value > 0:
                random_actions = torch.multinomial((~mask).to(values.dtype), 1).squeeze(
                    1
                )
                actions = torch.where(
                    torch.rand_like(values[:, 0]) < value, random_actions, actions
                )
        return actions.reshape(-1, 1)

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        ids = self._space_ids(available_action_space)
        state = (
            subjective_state.unsqueeze(0)
            if subjective_state.ndim == 1
            else subjective_state
        )
        if state.shape[0] != 1:
            raise ValueError(
                "native act accepts one state; use act_batch for vector inference"
            )
        mask = torch.ones((1, self._num_actions), device=state.device, dtype=torch.bool)
        mask[:, ids] = False
        return self.act_batch(state, mask, exploit=exploit).reshape(1)

    @torch.no_grad()
    def rollout_targets(self, rollout: PQNRollout) -> torch.Tensor:
        self._check_components()
        active = ~rollout.terminated
        dtype = next(self._Q.parameters()).dtype
        values = torch.zeros_like(rollout.rewards, dtype=dtype)
        if active.any():
            states = rollout.next_states[active]
            q_values = self._all_values(states)
            mask = (
                torch.zeros_like(q_values, dtype=torch.bool)
                if rollout.next_unavailable_actions_mask is None
                else rollout.next_unavailable_actions_mask[active]
            )
            if mask.shape != q_values.shape or mask.all(-1).any():
                raise ValueError("each bootstrapped row needs an available next action")
            if not torch.isfinite(q_values[~mask]).all():
                raise ValueError("nonfinite next-action values")
            values[active] = q_values.masked_fill(mask, -torch.inf).max(-1).values
        return q_lambda_returns(
            rollout.rewards.to(dtype=values.dtype),
            values,
            rollout.terminated,
            rollout.truncated,
            self._discount_factor,
            self._trace_decay,
        )

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        self._check_components()
        return batch

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        self._check_components()
        if not isinstance(batch, PQNTransitionBatch):
            raise TypeError(
                "PQN requires fixed chronological Q(lambda) targets, not replay TD samples"
            )
        if (
            len(batch) == 0
            or batch.returns.shape != (len(batch),)
            or batch.cost is not None
            or batch.weight is not None
        ):
            raise ValueError("invalid PQN batch or unsupported cost/importance weights")
        targets = batch.returns.detach()
        predictions = self._Q.get_q_values(batch.state, batch.action)
        loss = (predictions - targets).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite PQN loss")
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._max_grad_norm is not None:
            parameters = [
                p for group in self._optimizer.param_groups for p in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(
                parameters, self._max_grad_norm, error_if_nonfinite=True
            )
        self._optimizer.step()
        self._training_steps += 1
        return dict(loss=loss.item(), mean_q=predictions.detach().mean().item())

    def learn_rollout(self, rollout: PQNRollout) -> dict[str, Any]:
        targets = self.rollout_targets(rollout).reshape(-1)
        states = rollout.states.reshape(-1, rollout.states.shape[-1])
        actions = rollout.actions.reshape(-1, 1)
        reports: dict[str, list[float]] = {}
        before = self._training_steps
        for _ in range(self._training_rounds):
            permutation = torch.randperm(len(rollout), device=states.device)
            for start in range(0, len(rollout), self._batch_size):
                rows = permutation[start : start + self._batch_size]
                batch = PQNTransitionBatch(
                    state=states[rows],
                    action=actions[rows],
                    reward=rollout.rewards.reshape(-1)[rows],
                    terminated=rollout.terminated.reshape(-1)[rows],
                    truncated=rollout.truncated.reshape(-1)[rows],
                    returns=targets[rows],
                )
                for key, value in self.learn_batch(batch).items():
                    reports.setdefault(key, []).append(value)
        return dict(
            **reports,
            optimizer_updates=self._training_steps - before,
            transitions=len(rollout),
        )

    def learn(self, replay_buffer: ReplayBuffer) -> dict[str, Any]:
        if not isinstance(replay_buffer, PQNRolloutBuffer):
            raise TypeError(
                "PQN requires a consuming fresh-rollout buffer, not persistent replay"
            )
        if len(replay_buffer) == 0:
            return {}
        result = self.learn_rollout(replay_buffer.as_rollout())
        replay_buffer.clear()
        return result

    def get_extra_state(self) -> dict[str, Any]:
        return dict(
            version=1,
            config=copy.deepcopy(self._checkpoint_config),
            optimizer=copy.deepcopy(self._optimizer.state_dict()),
            training_steps=self._training_steps,
            environment_steps=self._environment_steps,
        )

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1 or state.get("config") != self._checkpoint_config:
            raise ValueError("checkpoint configuration must match ParallelQLearning")
        self._optimizer.load_state_dict(state["optimizer"])
        self._training_steps = int(state["training_steps"])
        self._environment_steps = int(state["environment_steps"])
