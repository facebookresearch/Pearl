# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""An experimental quantile-based Rainbow composition for fixed discrete actions.

This is a QR-DQN variant, not a reproduction of Rainbow's C51 agent. Replay
priorities use the unweighted per-transition quantile Huber loss.
"""

import copy
import math
from functools import partial
from typing import Any, cast

import torch
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.common.noisy_linear import NoisyLinear
from pearl.neural_networks.common.utils import update_target_network
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QuantileQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.rainbow_q_value_network import (
    RainbowQValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.wrappers.warmup import Warmup
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.quantile_regression_deep_q_learning import (
    QuantileRegressionDeepQLearning,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.prioritized_n_step_replay_buffer import (
    PrioritizedNStepReplayBuffer,
    PrioritizedNStepTransitionBatch,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.risk_sensitive_safety_modules import (
    RiskNeutralSafetyModule,
    RiskSensitiveSafetyModule,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class QuantileRainbow(QuantileRegressionDeepQLearning):
    """Compose Double QR-DQN, dueling heads, NoisyNet, and prioritized n-step replay.

    The catalog must contain scalar action IDs 0 through ``action_space.n - 1``.
    Available actions may be subsets of that fixed catalog. States may be any
    feature vectors; the learner has no task-specific feature assumptions.

    Use ``PrioritizedNStepReplayBuffer`` with the same ``discount_factor`` for
    the full composition. A regular ``TransitionBatch`` or replay buffer uses
    one-step targets and optional ``batch.weight``. Disabling ``double_q``,
    ``dueling`` or ``noisy`` supports component ablations. With ``noisy=False``
    the default exploration module is epsilon-greedy with epsilon 0.1.

    ``act(exploit=True)`` suppresses noisy layers for deterministic evaluation.
    One optimizer update increments the target-update counter once, including
    updates made through ``learn_batch``. State dicts contain model/optimizer
    state and this counter; replay contents and RNG states are not included.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: DiscreteActionSpace,
        hidden_dims: list[int] | None = None,
        num_quantiles: int = 51,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 32,
        target_update_freq: int = 200,
        soft_update_tau: float = 1.0,
        double_q: bool = True,
        dueling: bool = True,
        noisy: bool = True,
        std_init: float = 0.5,
        max_grad_norm: float | None = 10.0,
        exploration_module: EGreedyExploration | Warmup | None = None,
        action_representation_module: IdentityActionRepresentationModule | None = None,
    ) -> None:
        if not isinstance(action_space, DiscreteActionSpace):
            raise TypeError("QuantileRainbow requires a DiscreteActionSpace")
        if (
            action_space.actions_batch.dtype == torch.bool
            or action_space.actions_batch.is_complex()
            or action_space.action_dim != 1
            or not torch.equal(
                action_space.actions_batch.detach().cpu().reshape(-1),
                torch.arange(action_space.n),
            )
        ):
            raise ValueError(
                "action_space must be the ordered scalar catalog 0, ..., n-1"
            )
        if not math.isfinite(discount_factor) or not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be finite and in [0, 1]")
        for name, value in (
            ("target_update_freq", target_update_freq),
            ("training_rounds", training_rounds),
            ("batch_size", batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if (
            target_update_freq < 1
            or training_rounds < 1
            or batch_size == 0
            or batch_size < -1
        ):
            raise ValueError("invalid update frequency, training rounds, or batch size")
        if not math.isfinite(soft_update_tau) or not 0 < soft_update_tau <= 1:
            raise ValueError("soft_update_tau must be finite and in (0, 1]")
        if max_grad_norm is not None and (
            not math.isfinite(max_grad_norm) or max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be positive and finite")
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be nonnegative and finite")
        if exploration_module is not None and not (
            isinstance(exploration_module, EGreedyExploration)
            or (
                isinstance(exploration_module, Warmup)
                and isinstance(
                    exploration_module.exploration_module, EGreedyExploration
                )
            )
        ):
            raise TypeError(
                "this prototype supports epsilon-greedy exploration or its Warmup wrapper"
            )
        hidden_dims = [128, 128] if hidden_dims is None else list(hidden_dims)
        if action_representation_module is not None:
            self._check_identity_representation(
                action_representation_module, action_space.n
            )

        # The existing network_type hook calls its argument as a factory. Binding
        # the fixed catalog here avoids the different action_dim convention in
        # the upstream network_instance path; existing learner APIs are unchanged.
        network_factory = partial(
            RainbowQValueNetwork,
            num_actions=action_space.n,
            dueling=dueling,
            noisy=noisy,
            std_init=std_init,
        )
        if exploration_module is None:
            # epsilon=0 preserves the parent's single-observation exploration
            # interface while leaving exploration entirely to the noisy network.
            exploration_module = EGreedyExploration(0.0 if noisy else 0.1)
        super().__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            num_quantiles=num_quantiles,
            exploration_module=exploration_module,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            soft_update_tau=soft_update_tau,
            network_type=cast(type[QuantileQValueNetwork], network_factory),
            action_representation_module=action_representation_module,
        )
        self._double_q = double_q
        self._dueling = dueling
        self._noisy = noisy
        self._max_grad_norm = max_grad_norm
        self._num_actions = action_space.n
        self._Q_target.requires_grad_(False)
        self.safety_module = RiskNeutralSafetyModule()
        self._checkpoint_config = {
            "state_dim": state_dim,
            "num_actions": action_space.n,
            "hidden_dims": hidden_dims,
            "num_quantiles": num_quantiles,
            "discount_factor": discount_factor,
            "target_update_freq": target_update_freq,
            "soft_update_tau": soft_update_tau,
            "double_q": double_q,
            "dueling": dueling,
            "noisy": noisy,
            "std_init": std_init,
            "max_grad_norm": max_grad_norm,
        }

    def reset(self, action_space: ActionSpace) -> None:
        self._validate_available_actions(action_space)
        super().reset(action_space)

    def _validate_available_actions(self, action_space: ActionSpace) -> None:
        if (
            not isinstance(action_space, DiscreteActionSpace)
            or action_space.action_dim != 1
        ):
            raise ValueError(
                "available actions must be scalar IDs from the fixed catalog"
            )
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
            raise ValueError(
                "available actions must be unique IDs from the fixed catalog"
            )

    def _validate_action_representation(self) -> None:
        self._check_identity_representation(
            self.action_representation_module, self._num_actions
        )

    @staticmethod
    def _check_identity_representation(
        module: IdentityActionRepresentationModule, num_actions: int
    ) -> None:
        if (
            type(module) is not IdentityActionRepresentationModule
            or module.representation_dim != 1
            or module.max_number_actions != num_actions
            or any(p.requires_grad for p in module.parameters())
        ):
            raise ValueError(
                "QuantileRainbow requires a fixed identity action representation"
            )

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        self._validate_action_representation()
        # Follow PolicyLearner's history/action preprocessing contract, but do
        # not evaluate history on terminal placeholders or missing next states.
        active = self._bootstrap_discounts(batch) > 0
        batch.state = self._history_summarization_module(batch.state)
        with torch.no_grad():
            if active.any():
                if batch.next_state is None:
                    raise ValueError("bootstrapping requires next states")
                next_states = self._history_summarization_module(
                    batch.next_state[active]
                )
                batch.next_state = next_states.new_zeros(
                    (len(batch), *next_states.shape[1:])
                )
                batch.next_state[active] = next_states
            else:
                batch.next_state = None
        batch.action = self.action_representation_module(batch.action)
        if batch.next_action is not None:
            batch.next_action = self.action_representation_module(batch.next_action)
        if batch.curr_available_actions is not None:
            batch.curr_available_actions = self.action_representation_module(
                batch.curr_available_actions
            )
        if batch.next_available_actions is not None:
            batch.next_available_actions = self.action_representation_module(
                batch.next_available_actions
            )
        return batch

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        self._validate_available_actions(available_action_space)
        self._validate_action_representation()
        network = cast(RainbowQValueNetwork, self._Q)
        previous_noise = network.noise_enabled
        layer_noise = [
            (module, module.noise_enabled)
            for module in network.modules()
            if isinstance(module, NoisyLinear)
        ]
        try:
            network.enable_noise(self._noisy and not exploit)
            if self._noisy and not exploit:
                network.reset_noise()
            return super().act(
                subjective_state, available_action_space, exploit=exploit
            )
        finally:
            network.enable_noise(previous_noise)
            for module, enabled in layer_noise:
                module.enable_noise(enabled)

    def _bootstrap_discounts(self, batch: TransitionBatch) -> torch.Tensor:
        if isinstance(batch, PrioritizedNStepTransitionBatch):
            discounts = batch.bootstrap_discount.reshape(-1)
        else:
            discounts = self._discount_factor * (
                ~batch.terminated.bool()
            ).float().reshape(-1)
        if discounts.numel() != len(batch) or not torch.isfinite(discounts).all():
            raise ValueError("one finite bootstrap discount is required per transition")
        if (discounts < 0).any() or (discounts > 1).any():
            raise ValueError("bootstrap discounts must be in [0, 1]")
        if (discounts[batch.terminated.bool().reshape(-1)] != 0).any():
            raise ValueError(
                "true terminal transitions must have zero bootstrap discount"
            )
        return discounts

    @torch.no_grad()
    def _get_next_state_quantiles(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        result = batch.state.new_zeros((batch_size, self._num_quantiles))
        active = self._bootstrap_discounts(batch) > 0
        # Terminal rows can contain placeholders or no next state. Never pass
        # those rows through either network, avoiding NaN/Inf multiplied by zero.
        if not active.any():
            return result
        if batch.next_state is None or batch.next_available_actions is None:
            raise ValueError("bootstrapping requires next states and available actions")
        states = batch.next_state[active]
        actions = batch.next_available_actions[active]
        if actions.ndim != 3 or actions.shape[1] == 0:
            raise ValueError("bootstrapping requires a nonempty next-action axis")
        mask = batch.next_unavailable_actions_mask
        mask = (
            torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
            if (mask is None)
            else mask[active].bool()
        )
        if mask.shape != actions.shape[:2] or mask.all(dim=1).any():
            raise ValueError(
                "each bootstrapped state needs at least one available action"
            )

        # Replay padding is not a real action ID. Replace masked entries with a
        # harmless catalog ID before querying, then exclude them from selection.
        actions = actions.masked_fill(mask.unsqueeze(-1), 0)
        selector = self._Q if self._double_q else self._Q_target
        safety = self.safety_module
        if not isinstance(safety, RiskSensitiveSafetyModule):
            raise TypeError("QuantileRainbow requires a risk-sensitive safety module")
        values = safety.get_q_values_under_risk_metric(states, actions, selector)
        if not torch.isfinite(values[~mask]).all():
            raise ValueError("nonfinite values for available next actions")
        slots = values.masked_fill(mask, -torch.inf).argmax(dim=1)
        distributions = self._Q_target.get_q_value_distribution(states, actions)
        rows = torch.arange(states.shape[0], device=states.device)
        result[active] = distributions[rows, slots]
        return result

    @staticmethod
    def _per_transition_quantile_loss(
        prediction: torch.Tensor, target: torch.Tensor, midpoints: torch.Tensor
    ) -> torch.Tensor:
        # [batch, predicted quantile, target quantile]. Kappa is fixed to 1.
        delta = target.unsqueeze(1) - prediction.unsqueeze(2)
        absolute_delta = delta.abs()
        huber = torch.where(
            absolute_delta <= 1, 0.5 * delta.square(), absolute_delta - 0.5
        )
        tau = midpoints.reshape(1, -1, 1)
        weights = (tau - (delta.detach() < 0).to(delta.dtype)).abs()
        return (weights * huber).mean(dim=2).sum(dim=1)

    def _learn_batch_with_priorities(
        self, batch: TransitionBatch
    ) -> tuple[dict[str, float], torch.Tensor]:
        self._validate_action_representation()
        if len(batch) == 0:
            raise ValueError("cannot learn from an empty batch")
        cast(RainbowQValueNetwork, self._Q).reset_noise()
        cast(RainbowQValueNetwork, self._Q_target).reset_noise()
        prediction = self._Q.get_q_value_distribution(batch.state, batch.action)
        with torch.no_grad():
            target = batch.reward.reshape(-1, 1) + self._bootstrap_discounts(
                batch
            ).reshape(-1, 1) * self._get_next_state_quantiles(batch, len(batch))
        per_transition = self._per_transition_quantile_loss(
            prediction, target, self._Q.quantile_midpoints
        )
        weights = (
            torch.ones_like(per_transition)
            if batch.weight is None
            else batch.weight.reshape(-1)
        )
        if weights.shape != per_transition.shape or not torch.isfinite(weights).all():
            raise ValueError("one finite importance weight is required per transition")
        if (weights < 0).any():
            raise ValueError("importance weights cannot be negative")
        if not torch.isfinite(per_transition).all():
            raise ValueError("nonfinite quantile loss")
        loss = (weights * per_transition).mean()
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._max_grad_norm is not None:
            parameters = [
                p
                for group in self._optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(
                parameters, self._max_grad_norm, error_if_nonfinite=True
            )
        self._optimizer.step()
        self._training_steps += 1
        if self._training_steps % self._target_update_freq == 0:
            update_target_network(self._Q_target, self._Q, self._soft_update_tau)
        # Importance weights affect optimization, never the priorities themselves.
        priorities = per_transition.detach()
        return {
            "loss": loss.item(),
            "mean_priority": priorities.mean().item(),
        }, priorities

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        report, _ = self._learn_batch_with_priorities(batch)
        return report

    def learn(self, replay_buffer: ReplayBuffer) -> dict[str, Any]:
        if len(replay_buffer) == 0:
            return {}
        prioritized = isinstance(replay_buffer, PrioritizedNStepReplayBuffer)
        if prioritized and not math.isclose(
            replay_buffer.discount_factor,
            self._discount_factor,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("replay and learner must use the same discount_factor")
        batch_size = (
            len(replay_buffer)
            if self._batch_size == -1
            else min(self._batch_size, len(replay_buffer))
        )
        report: dict[str, list[float]] = {}
        for _ in range(self._training_rounds):
            batch = replay_buffer.sample(batch_size)
            if not isinstance(batch, TransitionBatch):
                raise TypeError("QuantileRainbow requires TransitionBatch samples")
            if prioritized and not isinstance(batch, PrioritizedNStepTransitionBatch):
                raise TypeError(
                    "prioritized replay requires indices and bootstrap discounts"
                )
            batch = self.preprocess_batch(batch)
            metrics, priorities = self._learn_batch_with_priorities(batch)
            if prioritized:
                replay_buffer.update_priorities(batch.indices, priorities)
            for name, value in metrics.items():
                report.setdefault(name, []).append(value)
        return report

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": copy.deepcopy(self._checkpoint_config),
            "training_steps": self._training_steps,
            "optimizer": copy.deepcopy(self._optimizer.state_dict()),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1 or state.get("config") != self._checkpoint_config:
            raise ValueError("checkpoint must match the QuantileRainbow configuration")
        self._training_steps = int(state["training_steps"])
        self._optimizer.load_state_dict(state["optimizer"])

    def compare(self, other: PolicyLearner) -> str:
        if not isinstance(other, QuantileRainbow):
            return "other is not an instance of QuantileRainbow"
        differences = [super().compare(other).strip()]
        if self._checkpoint_config != other._checkpoint_config:
            differences.append("QuantileRainbow configuration differs")
        return "\n".join(item for item in differences if item)
