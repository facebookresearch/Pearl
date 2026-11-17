# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Truncated Quantile Critics (Kuznetsov et al., arXiv:2005.04269)."""

import copy
import math
from contextlib import contextmanager
from typing import Any

import torch
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.common.utils import (
    update_target_network,
    xavier_init_weights,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    action_scaling,
    GaussianActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.quantile_critic import (
    QuantileCriticEnsemble,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.risk_sensitive_safety_modules import RiskSensitiveSafetyModule
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from torch import nn


@contextmanager
def _frozen_parameters(module: nn.Module):
    previous = [parameter.requires_grad for parameter in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(module.parameters(), previous):
            parameter.requires_grad_(enabled)


def truncated_quantiles(quantiles: torch.Tensor, drop_per_critic: int) -> torch.Tensor:
    """Pool, sort, then discard the largest N*drop_per_critic target atoms."""
    if quantiles.ndim != 3:
        raise ValueError("quantiles must have shape [batch, critics, quantiles]")
    if isinstance(drop_per_critic, bool) or not isinstance(drop_per_critic, int):
        raise ValueError("drop_per_critic must be an integer")
    if not 0 <= drop_per_critic < quantiles.shape[2] or quantiles.shape[1] == 0:
        raise ValueError(
            "truncation must leave at least one quantile per critic on average"
        )
    if not torch.isfinite(quantiles).all():
        raise ValueError("target quantiles must be finite")
    keep = quantiles.shape[1] * (quantiles.shape[2] - drop_per_critic)
    return quantiles.flatten(1).sort(dim=1).values[:, :keep]


def quantile_huber_loss(
    predictions: torch.Tensor, targets: torch.Tensor, midpoints: torch.Tensor
) -> torch.Tensor:
    """Per-transition loss, averaged over critics, predicted and target quantiles.

    Predictions [B,N,M], targets [B,K], and fractions [M]. Targets and the
    sign indicator are detached; kappa=1, as in TQC equation14.
    """
    if (
        predictions.ndim != 3
        or targets.ndim != 2
        or predictions.shape[0] != targets.shape[0]
    ):
        raise ValueError(
            "expected predictions [batch, critics, quantiles] and targets [batch, atoms]"
        )
    if midpoints.numel() != predictions.shape[2] or targets.shape[1] == 0:
        raise ValueError(
            "fractions must match prediction quantiles and target atoms must be nonempty"
        )
    delta = targets.detach()[:, None, None, :] - predictions[:, :, :, None]
    absolute = delta.abs()
    huber = torch.where(absolute <= 1, 0.5 * delta.square(), absolute - 0.5)
    weights = (
        midpoints.reshape(1, 1, -1, 1) - (delta.detach() < 0).to(delta.dtype)
    ).abs()
    return (weights * huber).mean(dim=(1, 2, 3))


class TruncatedQuantileCritics(ContinuousSoftActorCritic):
    """Continuous SAC with quantile critics and truncated value targets.

    Target predictions are pooled before the largest ones are removed. The
    actor uses the mean over all critics and quantiles.

    Actions use the environment's Box units. By default, entropy is measured
    in normalized [-1, 1] action units. Set
    entropy_in_normalized_coordinates=False to use native SAC's physical
    action density instead.

    Native actor, critic and history optimizer hooks are retained. Each
    learn_batch call updates the critic, actor and optional temperature.
    Checkpoints include optimizers and counters, but not replay, random
    generator state or live environments. Distributional risk modules are
    not supported by this learner.
    """

    def __init__(
        self,
        action_space: BoxActionSpace,
        state_dim: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        history_summarization_learning_rate: float = 3e-4,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        entropy_coef: float = 1.0,
        entropy_autotune: bool = True,
        num_critics: int = 2,
        num_quantiles: int = 25,
        top_quantiles_to_drop_per_net: int = 2,
        target_entropy: float | None = None,
        entropy_in_normalized_coordinates: bool = True,
        optimizer: str = "Adam",
        max_grad_norm: float | None = None,
        action_representation_module: IdentityActionRepresentationModule | None = None,
        actor_network_instance: GaussianActorNetwork | None = None,
        critic_network_instance: QuantileCriticEnsemble | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        critic_optimizer: torch.optim.Optimizer | None = None,
        history_summarization_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if not isinstance(action_space, BoxActionSpace):
            raise TypeError("TQC requires BoxActionSpace")
        low, high = action_space.low, action_space.high
        if (
            low.ndim != 1
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not (high > low).all()
        ):
            raise ValueError(
                "action bounds must be finite, one-dimensional and strictly increasing"
            )
        for name, value in [
            ("state_dim", state_dim),
            ("num_critics", num_critics),
            ("num_quantiles", num_quantiles),
            ("training_rounds", training_rounds),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size == 0
            or batch_size < -1
        ):
            raise ValueError("batch_size must be a positive integer or -1")
        if (
            isinstance(top_quantiles_to_drop_per_net, bool)
            or not isinstance(top_quantiles_to_drop_per_net, int)
            or not 0 <= top_quantiles_to_drop_per_net < num_quantiles
        ):
            raise ValueError(
                "top_quantiles_to_drop_per_net must be an integer in [0,num_quantiles)"
            )
        for name, value in [
            ("actor_learning_rate", actor_learning_rate),
            ("critic_learning_rate", critic_learning_rate),
            (
                "history_summarization_learning_rate",
                history_summarization_learning_rate,
            ),
        ]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(discount_factor) or not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be finite and in [0,1]")
        if (
            not math.isfinite(critic_soft_update_tau)
            or not 0 < critic_soft_update_tau <= 1
        ):
            raise ValueError("critic_soft_update_tau must be finite and in (0,1]")
        if (
            not math.isfinite(entropy_coef)
            or entropy_coef < 0
            or (entropy_autotune and entropy_coef <= 0)
        ):
            raise ValueError("entropy_coef must be finite and positive when autotuning")
        if target_entropy is not None and not math.isfinite(target_entropy):
            raise ValueError("target_entropy must be finite")
        if max_grad_norm is not None and (
            not math.isfinite(max_grad_norm) or max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")
        if optimizer not in ("Adam", "AdamW"):
            raise ValueError("optimizer must be Adam or AdamW")
        if action_representation_module is not None and (
            type(action_representation_module) is not IdentityActionRepresentationModule
            or action_representation_module.representation_dim
            != action_space.action_dim
            or any(p.requires_grad for p in action_representation_module.parameters())
        ):
            raise ValueError(
                "TQC currently requires fixed identity action representation"
            )
        actor_hidden_dims = (
            [256, 256] if actor_hidden_dims is None else list(actor_hidden_dims)
        )
        critic_hidden_dims = (
            [256, 256] if critic_hidden_dims is None else list(critic_hidden_dims)
        )
        actor = actor_network_instance or GaussianActorNetwork(
            state_dim, actor_hidden_dims, action_space.action_dim, action_space
        )
        if not isinstance(actor, GaussianActorNetwork):
            raise TypeError(
                "actor_network_instance must implement "
                "GaussianActorNetwork's scaling/density contract"
            )
        first_linear = next(
            m for m in actor._model.modules() if isinstance(m, nn.Linear)
        )
        if (
            first_linear.in_features != state_dim
            or actor.fc_mu.out_features != action_space.action_dim
            or not torch.equal(actor._action_space.low.cpu(), low.cpu())
            or not torch.equal(actor._action_space.high.cpu(), high.cpu())
        ):
            raise ValueError(
                "actor dimensions and action bounds must match the TQC configuration"
            )
        critic = critic_network_instance or QuantileCriticEnsemble(
            state_dim,
            action_space.action_dim,
            critic_hidden_dims,
            num_critics,
            num_quantiles,
        )
        if critic_network_instance is None:
            critic.apply(xavier_init_weights)
        if not isinstance(critic, QuantileCriticEnsemble) or (
            critic.state_dim,
            critic.action_dim,
            critic.num_critics,
            critic.num_quantiles,
        ) != (state_dim, action_space.action_dim, num_critics, num_quantiles):
            raise ValueError(
                "critic network dimensions must match the TQC configuration"
            )
        optimizer_class = torch.optim.Adam if optimizer == "Adam" else torch.optim.AdamW
        actor_optimizer = actor_optimizer or optimizer_class(
            actor.parameters(), lr=actor_learning_rate
        )
        critic_optimizer = critic_optimizer or optimizer_class(
            critic.parameters(), lr=critic_learning_rate
        )
        for name, supplied, network in [
            ("actor", actor_optimizer, actor),
            ("critic", critic_optimizer, critic),
        ]:
            owned = [p for group in supplied.param_groups for p in group["params"]]
            expected = [p for p in network.parameters() if p.requires_grad]
            if len(owned) != len({id(p) for p in owned}) or {id(p) for p in owned} != {
                id(p) for p in expected
            }:
                raise ValueError(
                    name
                    + " optimizer must own exactly its network's trainable parameters"
                )
        super().__init__(
            action_space=action_space,
            state_dim=state_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            history_summarization_learning_rate=history_summarization_learning_rate,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            entropy_coef=entropy_coef,
            entropy_autotune=entropy_autotune,
            action_representation_module=action_representation_module,
            actor_network_instance=actor,
            critic_network_instance=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            history_summarization_optimizer=history_summarization_optimizer,
        )
        self._num_critics, self._num_quantiles = num_critics, num_quantiles
        self._drop_per_critic = top_quantiles_to_drop_per_net
        self._entropy_in_normalized_coordinates = entropy_in_normalized_coordinates
        self._max_grad_norm = max_grad_norm
        self._gradient_steps = 0
        self.register_buffer("_log_action_scale", ((high - low) / 2).log().sum())
        if entropy_autotune:
            with torch.no_grad():
                self._log_entropy.fill_(math.log(entropy_coef))
            self._entropy_optimizer = optimizer_class(
                [self._log_entropy], lr=critic_learning_rate
            )
            self._entropy_coef = self._log_entropy.exp().detach()
            self._target_entropy = torch.tensor(
                -float(action_space.action_dim)
                if target_entropy is None
                else target_entropy
            )
        elif target_entropy is not None:
            raise ValueError("target_entropy only applies when entropy_autotune=True")
        self._critic_target.requires_grad_(False)
        self._checkpoint_config = dict(
            state_dim=state_dim,
            action_low=low.tolist(),
            action_high=high.tolist(),
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            num_critics=num_critics,
            num_quantiles=num_quantiles,
            drop_per_critic=top_quantiles_to_drop_per_net,
            discount_factor=discount_factor,
            critic_soft_update_tau=critic_soft_update_tau,
            entropy_autotune=entropy_autotune,
            entropy_in_normalized_coordinates=entropy_in_normalized_coordinates,
            target_entropy=target_entropy,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
        )

    def _check_components(self) -> None:
        module = self.action_representation_module
        if (
            type(module) is not IdentityActionRepresentationModule
            or module.representation_dim != self._critic.action_dim
            or any(p.requires_grad for p in module.parameters())
        ):
            raise ValueError("TQC requires fixed identity action representation")
        if isinstance(getattr(self, "safety_module", None), RiskSensitiveSafetyModule):
            raise TypeError(
                "TQC bias truncation does not implement quantile risk-sensitive policy objectives"
            )

    def _policy_action_and_log_prob(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, log_prob = self._actor.sample_action(state, get_log_prob=True)
        if self._entropy_in_normalized_coordinates:
            log_prob = log_prob + self._log_action_scale
        return action.to(state), log_prob.to(state)

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        self._check_components()
        if (
            not isinstance(available_action_space, BoxActionSpace)
            or not torch.equal(
                available_action_space.low.cpu(), self._actor._action_space.low.cpu()
            )
            or not torch.equal(
                available_action_space.high.cpu(), self._actor._action_space.high.cpu()
            )
        ):
            raise ValueError("available Box bounds must match the actor's fixed bounds")
        if exploit:
            with torch.no_grad():
                mean, _ = self._actor(subjective_state)
                return action_scaling(available_action_space, mean.tanh()).to(
                    subjective_state
                )
        return (
            super()
            .act(subjective_state, available_action_space, exploit=False)
            .to(subjective_state)
        )

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        self._check_components()
        if hasattr(self.safety_module, "lambda_constraint"):
            if batch.cost is None:
                raise ValueError("reward-constrained safety requires batch costs")
            if batch.cost.numel() != batch.reward.numel():
                raise ValueError("one cost is required per reward row")
            batch.reward = (
                batch.reward
                - self.safety_module.lambda_constraint
                * batch.cost.reshape_as(batch.reward)
            )
        active = (~batch.terminated.bool().reshape(-1)) & (self._discount_factor > 0)
        batch.state = self._history_summarization_module(batch.state)
        with torch.no_grad():
            if active.any():
                if batch.next_state is None:
                    raise ValueError("nonterminal transitions require next states")
                states = self._history_summarization_module(batch.next_state[active])
                batch.next_state = states.new_zeros((len(batch), *states.shape[1:]))
                batch.next_state[active] = states
            else:
                batch.next_state = None
        batch.action = self.action_representation_module(batch.action).to(batch.state)
        return batch

    @torch.no_grad()
    def _target_quantiles(self, batch: TransitionBatch) -> torch.Tensor:
        keep = self._num_critics * (self._num_quantiles - self._drop_per_critic)
        reward = batch.reward.reshape(-1)
        if not torch.isfinite(reward).all():
            raise ValueError("rewards must be finite")
        target = reward[:, None].expand(-1, keep).clone()
        active = (~batch.terminated.bool().reshape(-1)) & (self._discount_factor > 0)
        if active.any():
            if batch.next_state is None:
                raise ValueError("nonterminal transitions require next states")
            next_actions, next_log_prob = self._policy_action_and_log_prob(
                batch.next_state[active]
            )
            next_quantiles = self._critic_target.get_q_value_distribution(
                batch.next_state[active], next_actions
            )
            retained = truncated_quantiles(next_quantiles, self._drop_per_critic)
            target[active] += self._discount_factor * (
                retained - self._entropy_coef.detach() * next_log_prob.reshape(-1, 1)
            )
        return target

    @torch.no_grad()
    def _get_next_state_expected_values(self, batch: TransitionBatch) -> torch.Tensor:
        if self._discount_factor == 0:
            return batch.reward.new_zeros(len(batch))
        return (self._target_quantiles(batch) - batch.reward.reshape(-1, 1)).mean(
            1
        ) / self._discount_factor

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        predictions = self._critic.get_q_value_distribution(batch.state, batch.action)
        targets = self._target_quantiles(batch)
        per_transition = quantile_huber_loss(
            predictions, targets, self._critic.quantile_midpoints
        )
        weights = (
            torch.ones_like(per_transition)
            if batch.weight is None
            else batch.weight.reshape(-1)
        )
        if (
            weights.shape != per_transition.shape
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError(
                "one finite nonnegative critic importance weight is required per transition"
            )
        return (weights * per_transition).mean()

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        action, log_prob = self._policy_action_and_log_prob(batch.state)
        self._action_batch_log_prob_cache = log_prob.detach()
        with _frozen_parameters(self._critic):
            q = self._critic.get_q_value_distribution(batch.state, action).mean(
                dim=(1, 2)
            )
        return (self._entropy_coef.detach() * log_prob.reshape(-1) - q).mean()

    def _clip_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        if self._max_grad_norm is not None:
            parameters = [
                p
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(
                parameters, self._max_grad_norm, error_if_nonfinite=True
            )

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        self._check_components()
        if len(batch) == 0:
            raise ValueError("cannot learn from an empty batch")
        history_optimizer = self._history_summarization_optimizer
        if history_optimizer is not None:
            history_optimizer.zero_grad(set_to_none=True)
        critic_loss = self._critic_loss(batch)
        if not torch.isfinite(critic_loss):
            raise ValueError("nonfinite TQC critic loss")
        self._critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward(retain_graph=True)
        self._clip_gradients(self._critic_optimizer)
        self._critic_optimizer.step()
        self._critic_optimizer.zero_grad(set_to_none=True)
        actor_loss = self._actor_loss(batch)
        if not torch.isfinite(actor_loss):
            raise ValueError("nonfinite TQC actor loss")
        self._actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_gradients(self._actor_optimizer)
        self._actor_optimizer.step()
        if history_optimizer is not None:
            self._clip_gradients(history_optimizer)
            history_optimizer.step()
        entropy_loss = None
        if self._entropy_autotune:
            entropy_loss = -(
                self._log_entropy
                * (self._action_batch_log_prob_cache + self._target_entropy).detach()
            ).mean()
            self._entropy_optimizer.zero_grad(set_to_none=True)
            entropy_loss.backward()
            self._entropy_optimizer.step()
            self._entropy_coef = self._log_entropy.exp().detach()
        update_target_network(
            self._critic_target, self._critic, self._critic_soft_update_tau
        )
        self._gradient_steps += 1
        report = dict(
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            entropy_coef=self._entropy_coef.item(),
            gradient_steps=self._gradient_steps,
        )
        if entropy_loss is not None:
            report["entropy_loss"] = entropy_loss.item()
        return report

    def get_extra_state(self) -> dict[str, Any]:
        state = super().get_extra_state()
        state.update(
            version=1,
            config=copy.deepcopy(self._checkpoint_config),
            training_steps=self._training_steps,
            gradient_steps=self._gradient_steps,
        )
        if self._entropy_autotune:
            state["entropy_optimizer"] = copy.deepcopy(
                self._entropy_optimizer.state_dict()
            )
        return state

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1 or state.get("config") != self._checkpoint_config:
            raise ValueError("checkpoint must match the TQC configuration")
        super().set_extra_state(state)
        self._training_steps = int(state["training_steps"])
        self._gradient_steps = int(state["gradient_steps"])
        if self._entropy_autotune:
            self._entropy_optimizer.load_state_dict(state["entropy_optimizer"])

    def compare(self, other: PolicyLearner) -> str:
        if not isinstance(other, TruncatedQuantileCritics):
            return "other is not TruncatedQuantileCritics"
        # Native SAC compare assumes _target_entropy exists even when disabled.
        differences = [PolicyLearner.compare(self, other)]
        if self._checkpoint_config != other._checkpoint_config:
            differences.append("TQC configuration differs")
        for name in ["_actor", "_critic", "_critic_target"]:
            if any(
                not torch.equal(a, b)
                for a, b in zip(
                    getattr(self, name).parameters(), getattr(other, name).parameters()
                )
            ):
                differences.append(name + " parameters differ")
        return "\n".join(p for p in differences if p)
