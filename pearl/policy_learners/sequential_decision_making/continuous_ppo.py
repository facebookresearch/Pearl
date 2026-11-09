# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Continuous PPO with behavior-time Gaussian likelihoods and explicit rollouts.

Paper: https://arxiv.org/abs/1707.06347. The clipped-action convention and
orthogonal Tanh networks follow the official CleanRL continuous PPO/RPO reference
configuration. Native PEARL ActorCriticBase, ActorNetwork, ValueNetwork,
TransitionBatch, ReplayBuffer and PearlAgent interfaces are used.

CleanRL reference: https://github.com/vwxyzjn/cleanrl/tree/
fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl. The full
third-party notice is in third_party/cleanrl_mit_notice.txt.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.identity_history_summarization_module import (
    IdentityHistorySummarizationModule,
)
from pearl.neural_networks.sequential_decision_making.gaussian_ppo_actor import (
    GaussianPPOActorNetwork,
    PPOValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.sequential_decision_making.ppo_base import (
    ProximalPolicyOptimizationBase,
    generalized_advantage_estimate,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.identity_safety_module import IdentitySafetyModule
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace


@dataclass(kw_only=True)
class ContinuousPPOTransitionBatch(TransitionBatch):
    latent_action: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    behavior_policy_version: int | None = None


class ContinuousProximalPolicyOptimization(ProximalPolicyOptimizationBase):
    """PPO for finite Box actions, including native serial or batched collection.

    Executed actions are clipped to Box bounds; the original Gaussian sample is
    retained for likelihood ratios. Generic replay cannot reconstruct that sample
    or behavior-time probabilities, so use ContinuousPPORolloutBuffer or an
    explicit ContinuousPPOTransitionBatch. Only identity history/action/safety
    modules are supported initially. Recurrent sequence training needs a separate
    rollout API and is rejected rather than flattened silently.

    Actor and value parameters share an Adam optimizer; arbitrary native
    optimizer-instance hooks are not exposed by this initial API. Save
    collector.snapshot() with the model to restore evaluation normalization.
    This does not save the environment, RNG or in-flight rollout.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: BoxActionSpace,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        trace_decay: float = 0.95,
        update_epochs: int = 10,
        batch_size: int = 64,
        clip_epsilon: float = 0.2,
        value_loss_coefficient: float = 0.5,
        entropy_coefficient: float = 0.0,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        clip_value_loss: bool = True,
        target_kl: float | None = None,
    ) -> None:
        if not isinstance(action_space, BoxActionSpace):
            raise TypeError("continuous PPO requires BoxActionSpace")
        for name, value in [
            ("update_epochs", update_epochs),
            ("batch_size", batch_size),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in [
            ("learning_rate", learning_rate),
            ("clip_epsilon", clip_epsilon),
            ("max_grad_norm", max_grad_norm),
            ("value_loss_coefficient", value_loss_coefficient),
            ("entropy_coefficient", entropy_coefficient),
        ]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            max_grad_norm == 0
            or not 0 <= discount_factor <= 1
            or not 0 <= trace_decay <= 1
        ):
            raise ValueError("invalid gradient limit, discount or trace decay")
        if target_kl is not None and (not math.isfinite(target_kl) or target_kl <= 0):
            raise ValueError("target_kl must be finite and positive")
        hidden_dims = [64, 64] if hidden_dims is None else list(hidden_dims)
        actor = GaussianPPOActorNetwork(
            state_dim, hidden_dims, action_space.action_dim, action_space
        )
        critic = PPOValueNetwork(state_dim, hidden_dims)
        super().__init__(
            exploration_module=NoExploration(),
            state_dim=state_dim,
            action_space=action_space,
            actor_network_instance=actor,
            critic_network_instance=critic,
            is_action_continuous=True,
            on_policy=True,
            training_rounds=update_epochs,
            batch_size=batch_size,
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate,
        )
        # ActorCriticBase applies its default initialization even to injected
        # networks. Restore the explicitly declared PPO orthogonal initialization.
        actor.reset_parameters()
        critic.reset_parameters()
        self._optimizer = torch.optim.Adam(
            [*actor.parameters(), *critic.parameters()], lr=learning_rate, eps=1e-5
        )
        self._actor_optimizer = self._optimizer
        self._critic_optimizer = self._optimizer
        self._discount_factor, self._trace_decay = discount_factor, trace_decay
        self._clip_epsilon, self._value_coefficient, self._entropy_coefficient = (
            clip_epsilon,
            value_loss_coefficient,
            entropy_coefficient,
        )
        self._max_grad_norm, self._normalize_advantages, self._clip_value_loss = (
            max_grad_norm,
            normalize_advantages,
            clip_value_loss,
        )
        self._target_kl = target_kl
        self._epsilon = clip_epsilon
        self._trace_decay_param = trace_decay
        self._entropy_bonus_scaling = entropy_coefficient
        self._rpo_alpha = 0.0
        self._policy_version = 0
        self._pending_action: dict[str, Any] | None = None
        self.safety_module = IdentitySafetyModule()
        self._checkpoint_config = dict(
            state_dim=state_dim,
            action_low=actor.action_low.tolist(),
            action_high=actor.action_high.tolist(),
            hidden_dims=hidden_dims,
            discount_factor=discount_factor,
            trace_decay=trace_decay,
            update_epochs=update_epochs,
            batch_size=batch_size,
            clip_epsilon=clip_epsilon,
            value_loss_coefficient=value_loss_coefficient,
            entropy_coefficient=entropy_coefficient,
            max_grad_norm=max_grad_norm,
            normalize_advantages=normalize_advantages,
            clip_value_loss=clip_value_loss,
            target_kl=target_kl,
            rpo_alpha=0.0,
        )

    def set_history_summarization_module(self, value) -> None:
        if type(value) is not IdentityHistorySummarizationModule:
            raise TypeError(
                "continuous PPO requires identity history; "
                "recurrent rollouts are unsupported"
            )
        self._history_summarization_module = value
        self._history_summarization_optimizer = None

    def _check_components(self) -> None:
        if type(self.safety_module) is not IdentitySafetyModule:
            raise TypeError(
                "continuous PPO currently supports IdentitySafetyModule only"
            )
        module = self.action_representation_module
        if (
            type(module) is not IdentityActionRepresentationModule
            or any(p.requires_grad for p in module.parameters())
            or module.representation_dim != self._actor.action_low.numel()
        ):
            raise TypeError(
                "continuous PPO requires fixed identity action representation"
            )

    def reset(self, action_space: ActionSpace) -> None:
        self._validate_action_space(action_space)
        super().reset(action_space)

    def _validate_action_space(self, space) -> None:
        if (
            not isinstance(space, BoxActionSpace)
            or not torch.equal(space.low.cpu(), self._actor.action_low.cpu())
            or not torch.equal(space.high.cpu(), self._actor.action_high.cpu())
        ):
            raise ValueError("available Box bounds must match the policy action space")

    @torch.no_grad()
    def sample_actions(
        self, state_batch: torch.Tensor, exploit: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched behavior sampling; RPO perturbations occur only during updates."""
        self._check_components()
        distribution = self._actor.distribution(state_batch)
        latent = distribution.mean if exploit else distribution.sample()
        return (
            self._actor.clip_action(latent),
            latent,
            distribution.log_prob(latent).sum(-1),
            self._critic(state_batch).squeeze(-1),
        )

    def act(
        self,
        subjective_state: torch.Tensor,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> torch.Tensor:
        # Every action invalidates previous behavior metadata, including exploit
        # calls whose clipped action can coincide with a prior stochastic one.
        self._pending_action = None
        self._validate_action_space(available_action_space)
        if subjective_state.ndim not in (1, 2) or (
            subjective_state.ndim == 2 and subjective_state.shape[0] != 1
        ):
            raise ValueError(
                "act expects one state; use sample_actions for batched collection"
            )
        state = subjective_state.reshape(1, -1)
        action, latent, log_prob, value = self.sample_actions(state, exploit)
        if not exploit:
            self._pending_action = dict(
                state=state[0].detach().cpu().clone(),
                action=action[0].detach().cpu().clone(),
                latent_action=latent[0].detach().cpu().clone(),
                old_log_prob=log_prob[0].detach().cpu().clone(),
                old_value=value[0].detach().cpu().clone(),
                policy_version=self._policy_version,
            )
        return action[0]

    def consume_action_metadata(self, state, action) -> dict[str, Any]:
        metadata = self._pending_action
        if metadata is None:
            raise RuntimeError(
                "rollout push requires an immediately preceding policy act"
            )
        if not torch.equal(
            torch.as_tensor(state).detach().cpu().reshape(-1), metadata["state"]
        ) or not torch.equal(
            torch.as_tensor(action).detach().cpu().reshape(-1), metadata["action"]
        ):
            raise ValueError("rollout state/action do not match behavior-time metadata")
        if metadata["policy_version"] != self._policy_version:
            raise ValueError("policy changed before transition was recorded")
        self._pending_action = None
        return metadata

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        self._check_components()
        if not isinstance(batch, ContinuousPPOTransitionBatch):
            raise TypeError("PPO needs behavior-time rollout metadata")
        return batch

    def _loss_components(
        self, batch: ContinuousPPOTransitionBatch
    ) -> dict[str, torch.Tensor]:
        if not isinstance(batch, ContinuousPPOTransitionBatch) or len(batch) == 0:
            raise TypeError("nonempty PPO rollout batch required")
        for name in [
            "state",
            "action",
            "latent_action",
            "old_log_prob",
            "old_value",
            "advantages",
            "returns",
        ]:
            if not torch.isfinite(getattr(batch, name)).all():
                raise ValueError(f"nonfinite {name}")
        if (
            batch.action.shape != batch.latent_action.shape
            or batch.action.ndim != 2
            or batch.action.shape[-1] != self._actor.action_low.numel()
            or any(
                getattr(batch, name).numel() != len(batch)
                for name in ["old_log_prob", "old_value", "advantages", "returns"]
            )
        ):
            raise ValueError("invalid rollout shape")
        if not torch.equal(batch.action, self._actor.clip_action(batch.latent_action)):
            raise ValueError(
                "executed actions must match clipped latent behavior samples"
            )
        distribution = self._actor.distribution(batch.state, self._rpo_alpha)
        log_prob = distribution.log_prob(batch.latent_action.detach()).sum(-1)
        log_ratio = log_prob - batch.old_log_prob.detach().reshape(-1)
        ratio = log_ratio.exp()
        advantage = batch.advantages.detach().reshape(-1)
        if self._normalize_advantages:
            scale = (
                advantage.std()
                if advantage.numel() > 1
                else torch.zeros_like(advantage[0])
            )
            advantage = (advantage - advantage.mean()) / (scale + 1e-8)
        policy_loss = self.clipped_surrogate_loss(ratio, advantage, self._clip_epsilon)
        value = self._critic(batch.state).squeeze(-1)
        target = batch.returns.detach().reshape(-1)
        value_loss = self.value_regression_loss(
            value,
            target,
            old_value=(
                batch.old_value.detach().reshape(-1) if self._clip_value_loss else None
            ),
            epsilon=self._clip_epsilon,
            half=True,
        )
        return dict(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=distribution.entropy().sum(-1).mean(),
            approx_kl=((ratio - 1) - log_ratio).detach().mean(),
            clip_fraction=((ratio - 1).abs() > self._clip_epsilon).float().mean(),
        )

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        terms = self._loss_components(batch)
        return terms["policy_loss"] - self._entropy_coefficient * terms["entropy"]

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        return self._loss_components(batch)["value_loss"]

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        self._check_components()
        terms = self._loss_components(batch)
        loss = (
            terms["policy_loss"]
            + self._value_coefficient * terms["value_loss"]
            - self._entropy_coefficient * terms["entropy"]
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite PPO loss")
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*self._actor.parameters(), *self._critic.parameters()],
            self._max_grad_norm,
            error_if_nonfinite=True,
        )
        self._optimizer.step()
        self._training_steps += 1
        self._policy_version += 1
        return {
            "loss": loss.item(),
            **{key: value.item() for key, value in terms.items()},
        }

    def learn(self, replay_buffer: ReplayBuffer) -> dict[str, Any]:
        if (
            not isinstance(replay_buffer, ContinuousPPORolloutBuffer)
            or replay_buffer.policy is not self
        ):
            raise TypeError(
                "use this policy's ContinuousPPORolloutBuffer, not generic replay"
            )
        if not len(replay_buffer):
            return {}
        if replay_buffer.policy_version != self._policy_version:
            raise ValueError("stale on-policy rollout")
        return self.learn_rollout(replay_buffer.full_batch())

    def learn_rollout(self, full: ContinuousPPOTransitionBatch) -> dict[str, Any]:
        """Update from one complete fresh scalar or vector rollout."""
        if (
            not isinstance(full, ContinuousPPOTransitionBatch)
            or full.behavior_policy_version is None
            or int(full.behavior_policy_version) != self._policy_version
        ):
            raise ValueError("a fresh complete on-policy rollout is required")
        report: dict[str, list[float]] = {}
        stop = False
        for _ in range(self._training_rounds):
            order = torch.randperm(len(full), device=full.state.device)
            for indices in order.split(self._batch_size):
                fields = {
                    name: getattr(full, name)[indices]
                    for name in [
                        "state",
                        "action",
                        "reward",
                        "terminated",
                        "truncated",
                        "latent_action",
                        "old_log_prob",
                        "old_value",
                        "advantages",
                        "returns",
                    ]
                }
                metrics = self.learn_batch(ContinuousPPOTransitionBatch(**fields))
                for key, value in metrics.items():
                    report.setdefault(key, []).append(value)
                if (
                    self._target_kl is not None
                    and metrics["approx_kl"] > self._target_kl
                ):
                    stop = True
                    break
            if stop:
                break
        return report

    def get_extra_state(self) -> dict[str, Any]:
        return dict(
            version=1,
            config=copy.deepcopy(self._checkpoint_config),
            optimizer=copy.deepcopy(self._optimizer.state_dict()),
            training_steps=self._training_steps,
            policy_version=self._policy_version,
        )

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1 or state.get("config") != self._checkpoint_config:
            raise ValueError("checkpoint configuration mismatch")
        self._optimizer.load_state_dict(state["optimizer"])
        self._training_steps = int(state["training_steps"])
        self._policy_version = int(state["policy_version"])
        self._pending_action = None

    def compare(self, other) -> str:
        if type(other) is not type(self):
            return "continuous PPO/RPO learner types differ"
        differences = [super().compare(other).strip()]
        if self._checkpoint_config != other._checkpoint_config:
            differences.append("continuous PPO/RPO configuration differs")
        return "\n".join(item for item in differences if item)


class ContinuousPPORolloutBuffer(ReplayBuffer):
    """CPU on-policy rollout storage bound to one native serial PearlAgent.

    Metadata is captured by the policy at act time, including the unclipped
    Gaussian action. Advantages are computed once for the complete sequence,
    before any minibatch updates. No random replay across policy versions.
    """

    def __init__(
        self, capacity: int, policy: ContinuousProximalPolicyOptimization
    ) -> None:
        super().__init__()
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("positive integer capacity required")
        self.capacity, self.policy = capacity, policy
        self.records: list[dict[str, Any]] = []
        self._device = torch.device("cpu")
        self._is_action_continuous = True
        self.policy_version: int | None = None

    @property
    def device_for_batches(self) -> torch.device:
        return self._device

    @device_for_batches.setter
    def device_for_batches(self, value) -> None:
        self._device = torch.device(value)

    def push(
        self,
        state,
        action,
        reward,
        terminated,
        truncated,
        curr_available_actions=None,
        next_state=None,
        next_available_actions=None,
        max_number_actions=None,
        cost=None,
    ) -> None:
        if cost is not None:
            raise ValueError("PPO rollout does not support auxiliary costs")
        if len(self.records) >= self.capacity:
            raise ValueError("rollout is full; update before collecting more")
        if next_state is None and not terminated:
            raise ValueError("next observation required for nonterminal bootstrapping")
        metadata = self.policy.consume_action_metadata(state, action)
        device = next(self.policy.parameters()).device
        with torch.no_grad():
            next_value = (
                torch.zeros((), device=device)
                if terminated
                else self.policy._critic(
                    torch.as_tensor(
                        next_state, dtype=torch.float32, device=device
                    ).reshape(1, -1)
                ).reshape(())
            )
        self.push_metadata(metadata, reward, terminated, truncated, next_value)

    def push_metadata(
        self, metadata, reward, terminated, truncated, next_value
    ) -> None:
        if len(self.records) >= self.capacity:
            raise ValueError("rollout full")
        if self.policy_version is None:
            self.policy_version = metadata["policy_version"]
        if (
            metadata["policy_version"] != self.policy_version
            or metadata["policy_version"] != self.policy._policy_version
        ):
            raise ValueError("mixed behavior policies in rollout")
        record = {
            key: (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in metadata.items()
        }
        record.update(
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            next_value=float(next_value),
        )
        self.records.append(record)

    def full_batch(self) -> ContinuousPPOTransitionBatch:
        if not self.records:
            raise ValueError("empty rollout")

        def stack(name):
            return torch.stack([torch.as_tensor(row[name]) for row in self.records]).to(
                self._device
            )

        reward = stack("reward").float()
        values, next_values = stack("old_value"), stack("next_value")
        term, trunc = stack("terminated").bool(), stack("truncated").bool()
        advantages, returns = generalized_advantage_estimate(
            reward[:, None],
            values[:, None],
            next_values[:, None],
            term[:, None],
            trunc[:, None],
            self.policy._discount_factor,
            self.policy._trace_decay,
        )
        return ContinuousPPOTransitionBatch(
            state=stack("state"),
            action=stack("action"),
            reward=reward,
            terminated=term,
            truncated=trunc,
            latent_action=stack("latent_action"),
            old_log_prob=stack("old_log_prob"),
            old_value=values,
            advantages=advantages[:, 0],
            returns=returns[:, 0],
            behavior_policy_version=self.policy_version,
        )

    def sample(self, batch_size: int) -> ContinuousPPOTransitionBatch:
        if batch_size != len(self):
            raise ValueError(
                "on-policy rollout must be prepared as one complete sequence"
            )
        return self.full_batch()

    def clear(self) -> None:
        self.records.clear()
        self.policy_version = None

    def __len__(self) -> int:
        return len(self.records)
