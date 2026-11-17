# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental C51 Rainbow composition for fixed, scalar discrete action IDs.

See Hessel et al., arXiv:1710.02298 and Bellemare et al., arXiv:1707.06887.
This vector-observation implementation is not an Atari reproduction.
"""

import copy
import math
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
from pearl.neural_networks.sequential_decision_making.categorical_rainbow_q_value_network import (
    CategoricalRainbowQValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.wrappers.warmup import Warmup
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.prioritized_n_step_replay_buffer import (
    PrioritizedNStepReplayBuffer,
    PrioritizedNStepTransitionBatch,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.identity_safety_module import IdentitySafetyModule
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


@torch.no_grad()
def categorical_projection(
    probabilities: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Project r + discount * Z onto an equally spaced C51 support.

    Endpoint clipping preserves all probability mass. An atom falling exactly
    on a support bin contributes its entire mass once, including when the
    bootstrap discount is zero. This function creates no target gradient.
    """
    if support.ndim != 1 or support.numel() < 2 or not torch.isfinite(support).all():
        raise ValueError("support must be a finite vector of at least two atoms")
    spacing = support[1] - support[0]
    if spacing <= 0 or not torch.allclose(
        support[1:] - support[:-1],
        spacing.expand(support.numel() - 1),
        rtol=1e-4,
        atol=1e-6,
    ):
        raise ValueError("support must be equally spaced and strictly increasing")
    if probabilities.ndim != 2 or probabilities.shape[1] != support.numel():
        raise ValueError("probabilities must have shape [batch, atoms]")
    if (
        not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not torch.allclose(
            probabilities.sum(-1), torch.ones_like(probabilities[:, 0]), atol=1e-5
        )
    ):
        raise ValueError(
            "each probability row must be finite, nonnegative and sum to one"
        )
    rewards, discounts = rewards.reshape(-1), discounts.reshape(-1)
    if (
        rewards.numel() != probabilities.shape[0]
        or discounts.numel() != probabilities.shape[0]
    ):
        raise ValueError("one reward and discount are required per distribution")
    if (
        not torch.isfinite(rewards).all()
        or not torch.isfinite(discounts).all()
        or (discounts < 0).any()
        or (discounts > 1).any()
    ):
        raise ValueError("rewards must be finite and discounts must be in [0, 1]")
    target_atoms = rewards[:, None] + discounts[:, None] * support
    positions = (
        (target_atoms.clamp(support[0], support[-1]) - support[0]) / spacing
    ).clamp(0, support.numel() - 1)
    lower, upper = positions.floor().long(), positions.ceil().long()
    result = torch.zeros_like(probabilities)
    result.scatter_add_(
        1, lower, probabilities * (upper - positions + (lower == upper))
    )
    result.scatter_add_(1, upper, probabilities * (positions - lower))
    return result


class CategoricalRainbow(DeepQLearning):
    """Double C51, dueling, factorized NoisyNet, and prioritized n-step replay.

    Support bounds must match the task's discounted return scale; clipping mass
    is reported by learning. Priorities are the unweighted categorical KL loss,
    as in the Rainbow paper. Importance weights affect mean cross entropy only.
    Update frequency is measured in optimizer steps, not environment frames.

    PEARL's quantile-specific distributional safety modules are unsupported.
    The native QValueNetwork interface supplies probability-weighted expected
    values, using IdentitySafetyModule for reward-only risk-neutral behavior.
    Checkpoints contain model, optimizer and update counter, not replay/RNG/
    environment or exploration-schedule state; they are not exact resumes.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: DiscreteActionSpace,
        hidden_dims: list[int] | None = None,
        num_atoms: int = 51,
        value_min: float = -10.0,
        value_max: float = 10.0,
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
            raise TypeError("CategoricalRainbow requires DiscreteActionSpace")
        ids = action_space.actions_batch
        if (
            action_space.action_dim != 1
            or ids.dtype == torch.bool
            or ids.is_complex()
            or not torch.equal(
                ids.detach().cpu().reshape(-1), torch.arange(action_space.n)
            )
        ):
            raise ValueError(
                "action_space must be the ordered scalar catalog 0, ..., n-1"
            )
        if not math.isfinite(discount_factor) or not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be finite and in [0, 1]")
        for name, value in [
            ("target_update_freq", target_update_freq),
            ("training_rounds", training_rounds),
            ("batch_size", batch_size),
        ]:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if (
            target_update_freq < 1
            or training_rounds < 1
            or batch_size == 0
            or batch_size < -1
        ):
            raise ValueError("invalid update frequency, training rounds or batch size")
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if not math.isfinite(soft_update_tau) or not 0 < soft_update_tau <= 1:
            raise ValueError("soft_update_tau must be in (0, 1]")
        if max_grad_norm is not None and (
            not math.isfinite(max_grad_norm) or max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")
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
                "supported exploration is EGreedyExploration or its Warmup wrapper"
            )
        hidden_dims = [128, 128] if hidden_dims is None else list(hidden_dims)
        if action_representation_module is not None:
            self._check_identity(action_representation_module, action_space.n)
        network = CategoricalRainbowQValueNetwork(
            state_dim,
            action_space.n,
            hidden_dims,
            num_atoms,
            value_min,
            value_max,
            dueling,
            noisy,
            std_init,
        )
        super().__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            exploration_module=exploration_module
            or EGreedyExploration(0.0 if noisy else 0.1),
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            soft_update_tau=soft_update_tau,
            network_instance=network,
            action_representation_module=action_representation_module,
        )
        self._num_actions, self._num_atoms = action_space.n, num_atoms
        self._double_q, self._noisy, self._max_grad_norm = (
            double_q,
            noisy,
            max_grad_norm,
        )
        self._Q_target.requires_grad_(False)
        self.safety_module = IdentitySafetyModule()
        self._checkpoint_config = dict(
            state_dim=state_dim,
            num_actions=action_space.n,
            hidden_dims=hidden_dims,
            num_atoms=num_atoms,
            value_min=value_min,
            value_max=value_max,
            discount_factor=discount_factor,
            target_update_freq=target_update_freq,
            soft_update_tau=soft_update_tau,
            double_q=double_q,
            dueling=dueling,
            noisy=noisy,
            std_init=std_init,
            max_grad_norm=max_grad_norm,
        )

    @staticmethod
    def _check_identity(
        module: IdentityActionRepresentationModule, num_actions: int
    ) -> None:
        if (
            type(module) is not IdentityActionRepresentationModule
            or module.representation_dim != 1
            or module.max_number_actions != num_actions
            or any(p.requires_grad for p in module.parameters())
        ):
            raise ValueError(
                "CategoricalRainbow requires fixed identity action representation"
            )

    def _check_components(self) -> None:
        self._check_identity(self.action_representation_module, self._num_actions)
        if type(self.safety_module) is not IdentitySafetyModule:
            raise TypeError(
                "CategoricalRainbow currently supports only IdentitySafetyModule"
            )

    def _validate_actions(self, space: ActionSpace) -> None:
        if not isinstance(space, DiscreteActionSpace) or space.action_dim != 1:
            raise ValueError("available actions must be scalar catalog IDs")
        ids = space.actions_batch.reshape(-1)
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
                "available actions must be unique IDs within the fixed catalog"
            )

    def reset(self, action_space: ActionSpace) -> None:
        self._validate_actions(action_space)
        super().reset(action_space)

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        self._validate_actions(available_action_space)
        self._check_components()
        network = cast(CategoricalRainbowQValueNetwork, self._Q)
        previous = network.noise_enabled
        switches = [
            (m, m.noise_enabled)
            for m in network.modules()
            if isinstance(m, NoisyLinear)
        ]
        try:
            network.enable_noise(self._noisy and not exploit)
            if self._noisy and not exploit:
                network.reset_noise()
            return super().act(
                subjective_state, available_action_space, exploit=exploit
            )
        finally:
            network.enable_noise(previous)
            for module, enabled in switches:
                module.enable_noise(enabled)

    def _bootstrap_discounts(self, batch: TransitionBatch) -> torch.Tensor:
        discounts = (
            batch.bootstrap_discount.reshape(-1)
            if isinstance(batch, PrioritizedNStepTransitionBatch)
            else self._discount_factor * (~batch.terminated.bool()).float().reshape(-1)
        )
        if (
            discounts.numel() != len(batch)
            or not torch.isfinite(discounts).all()
            or (discounts < 0).any()
            or (discounts > 1).any()
            or (discounts[batch.terminated.bool().reshape(-1)] != 0).any()
        ):
            raise ValueError(
                "one finite discount in [0, 1] required per row, zero at true terminals"
            )
        return discounts

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        self._check_components()
        active = self._bootstrap_discounts(batch) > 0
        batch.state = self._history_summarization_module(batch.state)
        with torch.no_grad():
            if active.any():
                if batch.next_state is None:
                    raise ValueError("bootstrapping requires next states")
                states = self._history_summarization_module(batch.next_state[active])
                batch.next_state = states.new_zeros((len(batch), *states.shape[1:]))
                batch.next_state[active] = states
            else:
                batch.next_state = None
        for name in [
            "action",
            "next_action",
            "curr_available_actions",
            "next_available_actions",
        ]:
            value = getattr(batch, name)
            if value is not None:
                setattr(batch, name, self.action_representation_module(value))
        return batch

    @torch.no_grad()
    def _next_probabilities(self, batch: TransitionBatch) -> torch.Tensor:
        result = batch.state.new_zeros((len(batch), self._num_atoms))
        result[:, 0] = 1  # arbitrary normalized terminal distribution; discount is zero
        active = self._bootstrap_discounts(batch) > 0
        if not active.any():
            return result
        if batch.next_state is None or batch.next_available_actions is None:
            raise ValueError("bootstrapping requires next states and available actions")
        states, actions = batch.next_state[active], batch.next_available_actions[active]
        if actions.ndim != 3 or actions.shape[1] == 0:
            raise ValueError("bootstrapping requires a nonempty next-action axis")
        mask = (
            torch.zeros(actions.shape[:2], device=actions.device, dtype=torch.bool)
            if batch.next_unavailable_actions_mask is None
            else batch.next_unavailable_actions_mask[active].bool()
        )
        if mask.shape != actions.shape[:2] or mask.all(1).any():
            raise ValueError(
                "each bootstrapped row needs at least one available action"
            )
        actions = actions.masked_fill(mask.unsqueeze(-1), 0)
        selector = self._Q if self._double_q else self._Q_target
        values = selector.get_q_values(states, actions)
        if not torch.isfinite(values[~mask]).all():
            raise ValueError("nonfinite next-action values")
        slots = values.masked_fill(mask, -torch.inf).argmax(1)
        selected = actions[torch.arange(states.shape[0], device=states.device), slots]
        result[active] = self._Q_target.get_probabilities(states, selected)
        return result

    def get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        return (self._next_probabilities(batch) * self._Q_target.support).sum(-1) * (
            self._bootstrap_discounts(batch) > 0
        )

    def _learn_batch_with_priorities(
        self, batch: TransitionBatch
    ) -> tuple[dict[str, float], torch.Tensor]:
        self._check_components()
        if len(batch) == 0:
            raise ValueError("cannot learn from an empty batch")
        self._Q.reset_noise()
        self._Q_target.reset_noise()
        logits = self._Q.get_logits(batch.state, batch.action)
        with torch.no_grad():
            next_probabilities = self._next_probabilities(batch)
            discounts = self._bootstrap_discounts(batch)
            support = self._Q.support
            atoms = batch.reward.reshape(-1, 1) + discounts[:, None] * support
            clipped_mass = (
                (next_probabilities * ((atoms < support[0]) | (atoms > support[-1])))
                .sum(-1)
                .mean()
            )
            target = categorical_projection(
                next_probabilities, batch.reward, discounts, support
            )
        log_probabilities = logits.log_softmax(-1)
        per_transition = -(target * log_probabilities).sum(-1)
        priorities = (
            per_transition.detach() + torch.special.xlogy(target, target).sum(-1)
        ).clamp_min(0)
        weights = (
            torch.ones_like(per_transition)
            if batch.weight is None
            else batch.weight.reshape(-1)
        )
        if (
            weights.shape != per_transition.shape
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or not torch.isfinite(per_transition).all()
        ):
            raise ValueError(
                "loss and one nonnegative importance weight per row must be finite"
            )
        loss = (weights * per_transition).mean()
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._max_grad_norm is not None:
            parameters = [
                p
                for g in self._optimizer.param_groups
                for p in g["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(
                parameters, self._max_grad_norm, error_if_nonfinite=True
            )
        self._optimizer.step()
        self._training_steps += 1
        if self._training_steps % self._target_update_freq == 0:
            update_target_network(self._Q_target, self._Q, self._soft_update_tau)
        return (
            dict(
                loss=loss.item(),
                mean_priority=priorities.mean().item(),
                support_clipped_mass=clipped_mass.item(),
            ),
            priorities,
        )

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        metrics, _ = self._learn_batch_with_priorities(batch)
        return metrics

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
            raise ValueError("replay and learner discount_factor must agree")
        size = (
            len(replay_buffer)
            if self._batch_size == -1
            else min(self._batch_size, len(replay_buffer))
        )
        report: dict[str, list[float]] = {}
        for _ in range(self._training_rounds):
            batch = replay_buffer.sample(size)
            if not isinstance(batch, TransitionBatch) or (
                prioritized and not isinstance(batch, PrioritizedNStepTransitionBatch)
            ):
                raise TypeError("replay sample must preserve transition metadata")
            batch = self.preprocess_batch(batch)
            metrics, priorities = self._learn_batch_with_priorities(batch)
            if prioritized:
                replay_buffer.update_priorities(batch.indices, priorities)
            for key, value in metrics.items():
                report.setdefault(key, []).append(value)
        return report

    def get_extra_state(self) -> dict[str, Any]:
        return dict(
            version=1,
            config=copy.deepcopy(self._checkpoint_config),
            training_steps=self._training_steps,
            optimizer=copy.deepcopy(self._optimizer.state_dict()),
        )

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1 or state.get("config") != self._checkpoint_config:
            raise ValueError("checkpoint must match CategoricalRainbow configuration")
        self._training_steps = int(state["training_steps"])
        self._optimizer.load_state_dict(state["optimizer"])

    def compare(self, other: PolicyLearner) -> str:
        if not isinstance(other, CategoricalRainbow):
            return "other is not an instance of CategoricalRainbow"
        differences = [super().compare(other).strip()]
        if self._checkpoint_config != other._checkpoint_config:
            differences.append("CategoricalRainbow configuration differs")
        return "\n".join(item for item in differences if item)
