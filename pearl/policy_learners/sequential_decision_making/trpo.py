# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Continuous Trust Region Policy Optimization (Schulman et al., 2015)."""

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.identity_history_summarization_module import (
    IdentityHistorySummarizationModule,
)
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.neural_networks.sequential_decision_making.gaussian_ppo_actor import (
    GaussianPPOActorNetwork,
    PPOValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.policy_learners.sequential_decision_making.continuous_ppo import (
    ContinuousPPORolloutBuffer,
    ContinuousPPOTransitionBatch,
)
from pearl.safety_modules.identity_safety_module import IdentitySafetyModule
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from pearl.utils.module_utils import state_dict_owners_have_similar_state_dict


def gaussian_kl(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
) -> torch.Tensor:
    """KL(old || new), summed over actions and averaged over states."""
    terms = (
        new_std.log()
        - old_std.log()
        + (old_std.square() + (old_mean - new_mean).square()) / (2 * new_std.square())
        - 0.5
    )
    return terms.sum(-1).mean()


@dataclass
class ConjugateGradientResult:
    solution: torch.Tensor
    iterations: int
    residual_norm: float
    converged: bool
    breakdown: bool


def conjugate_gradient(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    max_iterations: int = 10,
    tolerance: float = 1e-10,
) -> ConjugateGradientResult:
    """Solve a positive-definite system without forming its matrix."""
    if rhs.ndim != 1 or not torch.isfinite(rhs).all():
        raise ValueError("a finite vector is required")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("positive CG iteration limit and tolerance required")
    solution = torch.zeros_like(rhs)
    residual = rhs.detach().clone()
    direction = residual.clone()
    squared = residual.dot(residual)
    iterations, breakdown = 0, False
    for _ in range(max_iterations):
        if squared.sqrt().item() <= tolerance:
            break
        product = operator(direction).detach()
        if product.shape != rhs.shape:
            raise ValueError("CG operator changed the vector shape")
        curvature = direction.dot(product)
        if (
            not torch.isfinite(product).all()
            or not torch.isfinite(curvature)
            or curvature <= 0
        ):
            breakdown = True
            break
        alpha = squared / curvature
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_squared = residual.dot(residual)
        iterations += 1
        if not torch.isfinite(next_squared) or not torch.isfinite(solution).all():
            breakdown = True
            break
        direction = residual + (next_squared / squared) * direction
        squared = next_squared
    norm = float(residual.norm())
    return ConjugateGradientResult(
        solution.detach(),
        iterations,
        norm,
        not breakdown and norm <= tolerance,
        breakdown,
    )


class ContinuousTrustRegionPolicyOptimization(ActorCriticBase):
    """TRPO for finite Box actions and fresh Gaussian rollouts.

    Likelihoods use the sampled Gaussian action before Box clipping. The actor
    takes one trust-region step per rollout; a separate optimizer fits values.
    Identity history, action representation and safety modules are supported.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: BoxActionSpace,
        hidden_dims: list[int] | None = None,
        critic_learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        trace_decay: float = 0.95,
        critic_update_epochs: int = 10,
        batch_size: int = 64,
        target_kl: float = 0.01,
        cg_max_iterations: int = 10,
        cg_damping: float = 0.1,
        cg_tolerance: float = 1e-10,
        line_search_max_iterations: int = 10,
        line_search_shrink: float = 0.5,
        accept_ratio: float = 0.1,
        normalize_advantages: bool = True,
        critic_max_grad_norm: float | None = None,
    ) -> None:
        if not isinstance(action_space, BoxActionSpace):
            raise TypeError("continuous TRPO requires BoxActionSpace")
        for name, value in [
            ("critic_update_epochs", critic_update_epochs),
            ("batch_size", batch_size),
            ("cg_max_iterations", cg_max_iterations),
            ("line_search_max_iterations", line_search_max_iterations),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in [
            ("critic_learning_rate", critic_learning_rate),
            ("target_kl", target_kl),
            ("cg_tolerance", cg_tolerance),
        ]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(cg_damping)
            or cg_damping < 0
            or not 0 < line_search_shrink < 1
        ):
            raise ValueError("invalid damping or line-search shrink factor")
        if not math.isfinite(accept_ratio) or not 0 <= accept_ratio <= 1:
            raise ValueError("accept_ratio must lie in [0,1]")
        if not 0 <= discount_factor <= 1 or not 0 <= trace_decay <= 1:
            raise ValueError("discount and trace decay must lie in [0,1]")
        if critic_max_grad_norm is not None and (
            not math.isfinite(critic_max_grad_norm) or critic_max_grad_norm <= 0
        ):
            raise ValueError("critic gradient limit must be finite and positive")
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
            training_rounds=1,
            batch_size=batch_size,
            actor_learning_rate=0.0,
            critic_learning_rate=critic_learning_rate,
            discount_factor=discount_factor,
        )
        actor.reset_parameters()
        critic.reset_parameters()
        self._actor_optimizer = None
        self._critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=critic_learning_rate
        )
        self._trace_decay = trace_decay
        self._critic_epochs = critic_update_epochs
        self._target_kl, self._cg_iterations, self._cg_damping = (
            target_kl,
            cg_max_iterations,
            cg_damping,
        )
        self._cg_tolerance = cg_tolerance
        self._line_search_iterations, self._line_search_shrink = (
            line_search_max_iterations,
            line_search_shrink,
        )
        self._accept_ratio, self._normalize_advantages = (
            accept_ratio,
            normalize_advantages,
        )
        self._critic_max_grad_norm = critic_max_grad_norm
        self._policy_version = 0
        self._pending_action = None
        self._behavior_sampling_policy_version: int | None = None
        self._behavior_sampling_batch_size: int | None = None
        self._actor_attempts = self._actor_accepted = self._actor_rejected = (
            self._actor_numerical_failures
        ) = 0
        self._critic_updates = 0
        self._last_actor_rejection_reason = None
        self.safety_module = IdentitySafetyModule()
        self._checkpoint_config = dict(
            state_dim=state_dim,
            action_low=actor.action_low.tolist(),
            action_high=actor.action_high.tolist(),
            hidden_dims=hidden_dims,
            critic_learning_rate=critic_learning_rate,
            discount_factor=discount_factor,
            trace_decay=trace_decay,
            critic_update_epochs=critic_update_epochs,
            batch_size=batch_size,
            target_kl=target_kl,
            cg_max_iterations=cg_max_iterations,
            cg_damping=cg_damping,
            cg_tolerance=cg_tolerance,
            line_search_max_iterations=line_search_max_iterations,
            line_search_shrink=line_search_shrink,
            accept_ratio=accept_ratio,
            normalize_advantages=normalize_advantages,
            critic_max_grad_norm=critic_max_grad_norm,
        )

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        if type(value) is not IdentityHistorySummarizationModule:
            raise TypeError("continuous TRPO requires identity history")
        self._history_summarization_module = value
        self._history_summarization_optimizer = None

    def _check_components(self) -> None:
        module = self.action_representation_module
        if type(self.safety_module) is not IdentitySafetyModule:
            raise TypeError("continuous TRPO requires IdentitySafetyModule")
        if (
            type(module) is not IdentityActionRepresentationModule
            or any(p.requires_grad for p in module.parameters())
            or module.representation_dim != self._actor.action_low.numel()
        ):
            raise TypeError(
                "continuous TRPO requires fixed identity action representation"
            )

    def _validate_action_space(self, space: ActionSpace) -> None:
        if (
            not isinstance(space, BoxActionSpace)
            or not torch.equal(space.low.cpu(), self._actor.action_low.cpu())
            or not torch.equal(space.high.cpu(), self._actor.action_high.cpu())
        ):
            raise ValueError("available Box bounds must match the policy action space")

    def reset(self, action_space: ActionSpace) -> None:
        self._validate_action_space(action_space)
        super().reset(action_space)

    @torch.no_grad()
    def sample_actions(
        self, state_batch: torch.Tensor, exploit: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state_batch.ndim != 2 or state_batch.shape[0] < 1:
            raise ValueError("sample_actions expects a nonempty batch of states")
        self._check_components()
        distribution = self._actor.distribution(state_batch)
        latent = distribution.mean if exploit else distribution.sample()
        result = (
            self._actor.clip_action(latent),
            latent,
            distribution.log_prob(latent).sum(-1),
            self._critic(state_batch).squeeze(-1),
        )
        if not exploit:
            size = state_batch.shape[0]
            if self._behavior_sampling_policy_version != self._policy_version:
                self._behavior_sampling_policy_version = self._policy_version
                self._behavior_sampling_batch_size = size
            elif self._behavior_sampling_batch_size != size:
                self._behavior_sampling_batch_size = None
        return result

    def _clear_behavior_sampling_metadata(self) -> None:
        self._behavior_sampling_policy_version = None
        self._behavior_sampling_batch_size = None

    def act(
        self,
        subjective_state: torch.Tensor,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> torch.Tensor:
        self._pending_action = None
        self._validate_action_space(available_action_space)
        if subjective_state.ndim not in (1, 2) or (
            subjective_state.ndim == 2 and subjective_state.shape[0] != 1
        ):
            raise ValueError("act expects one state; use sample_actions for batches")
        state = subjective_state.reshape(1, -1)
        action, latent, log_prob, value = self.sample_actions(state, exploit)
        if not exploit:
            self._pending_action = {
                name: tensor[0].detach().cpu().clone()
                for name, tensor in dict(
                    state=state,
                    action=action,
                    latent_action=latent,
                    old_log_prob=log_prob,
                    old_value=value,
                ).items()
            }
            self._pending_action["policy_version"] = self._policy_version
        return action[0]

    def consume_action_metadata(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> dict[str, Any]:
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
            raise ValueError("rollout state/action do not match behavior metadata")
        if metadata["policy_version"] != self._policy_version:
            raise ValueError("policy changed before transition was recorded")
        self._pending_action = None
        return metadata

    def preprocess_batch(self, batch: TransitionBatch) -> ContinuousPPOTransitionBatch:
        self._check_components()
        if not isinstance(batch, ContinuousPPOTransitionBatch):
            raise TypeError("TRPO needs behavior-time rollout metadata")
        return batch

    def _validate_rollout(self, batch: ContinuousPPOTransitionBatch) -> None:
        self.preprocess_batch(batch)
        version = batch.behavior_policy_version
        if isinstance(version, torch.Tensor):
            if version.numel() != 1 or version.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise ValueError("rollout policy version must be an integer")
            version = int(version.item())
        if (
            len(batch) < 1
            or type(version) is not int
            or version != self._policy_version
        ):
            raise ValueError("a fresh complete on-policy rollout is required")
        n, a = len(batch), self._actor.action_low.numel()
        shapes = dict(
            state=(n, self._state_dim),
            action=(n, a),
            latent_action=(n, a),
            reward=(n,),
            old_log_prob=(n,),
            old_value=(n,),
            advantages=(n,),
            returns=(n,),
        )
        for name, shape in shapes.items():
            value = getattr(batch, name)
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError("invalid rollout field: " + name)
        if not torch.equal(batch.action, self._actor.clip_action(batch.latent_action)):
            raise ValueError("executed actions must match clipped Gaussian samples")
        if batch.cost is not None or batch.weight is not None:
            raise ValueError("TRPO requires an unweighted reward-only rollout")
        with torch.no_grad():
            log_prob = self._actor.get_action_log_prob(batch.state, batch.latent_action)
            if not torch.allclose(log_prob, batch.old_log_prob, atol=2e-5, rtol=2e-6):
                size = self._behavior_sampling_batch_size
                if (
                    self._behavior_sampling_policy_version != self._policy_version
                    or size is None
                    or len(batch) % size
                ):
                    raise ValueError(
                        "behavior likelihoods do not match the current actor"
                    )
                # Match collection's small batches and fresh tensor storage.
                log_prob = torch.cat(
                    [
                        self._actor.get_action_log_prob(
                            states.clone(memory_format=torch.contiguous_format),
                            actions.clone(memory_format=torch.contiguous_format),
                        )
                        for states, actions in zip(
                            batch.state.split(size), batch.latent_action.split(size)
                        )
                    ]
                )
                if not torch.allclose(
                    log_prob, batch.old_log_prob, atol=2e-5, rtol=2e-6
                ):
                    raise ValueError(
                        "behavior likelihoods do not match the current actor"
                    )

    def _actor_loss(self, batch: ContinuousPPOTransitionBatch) -> torch.Tensor:
        ratio = (
            self._actor.get_action_log_prob(batch.state, batch.latent_action)
            - batch.old_log_prob
        ).exp()
        return -(ratio * batch.advantages.detach()).mean()

    def _critic_loss(self, batch: ContinuousPPOTransitionBatch) -> torch.Tensor:
        return (
            (self._critic(batch.state).squeeze(-1) - batch.returns.detach())
            .square()
            .mean()
        )

    def fisher_vector_product(
        self,
        states: torch.Tensor,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        vector: torch.Tensor,
        damping: float | None = None,
    ) -> torch.Tensor:
        """Multiply by the joint Gaussian KL Hessian, with optional damping."""
        damping = self._cg_damping if damping is None else damping
        if not math.isfinite(damping) or damping < 0:
            raise ValueError("Fisher damping must be finite and nonnegative")
        parameters = list(self._actor.parameters())
        distribution = self._actor.distribution(states.detach())
        kl = gaussian_kl(
            old_mean.detach(), old_std.detach(), distribution.mean, distribution.stddev
        )
        first = torch.autograd.grad(kl, parameters, create_graph=True)
        flat = torch.cat([value.reshape(-1) for value in first])
        if vector.shape != flat.shape or not torch.isfinite(vector).all():
            raise ValueError("invalid Fisher vector")
        second = torch.autograd.grad(flat.dot(vector.detach()), parameters)
        result = torch.cat([value.reshape(-1) for value in second]).detach()
        return result + damping * vector.detach()

    @torch.no_grad()
    def _set_actor_parameters(self, flat: torch.Tensor) -> None:
        offset = 0
        for parameter in self._actor.parameters():
            count = parameter.numel()
            parameter.copy_(flat[offset : offset + count].view_as(parameter))
            offset += count
        if offset != flat.numel():
            raise ValueError("actor parameter vector has the wrong size")

    def _accept_actor_candidate(
        self, improvement: float, kl: float, expected_improvement: float
    ) -> bool:
        return (
            -1e-7 <= kl <= self._target_kl
            and improvement > 0
            and improvement >= self._accept_ratio * expected_improvement
        )

    def _actor_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float]:
        parameters = list(self._actor.parameters())
        original = torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
        with torch.no_grad():
            old = self._actor.distribution(states)
            old_mean, old_std = old.mean.detach().clone(), old.stddev.detach().clone()
        accepted, reason = False, "line_search"
        metrics = dict(
            actor_step_accepted=0.0,
            actor_step_rejected=1.0,
            actor_numerical_failure=0.0,
            cg_breakdown=0.0,
            cg_iterations=0.0,
            cg_residual_norm=0.0,
            cg_converged=0.0,
            line_search_backtracks=0.0,
            line_search_nonfinite_candidates=0.0,
            surrogate_improvement=0.0,
            expected_improvement=0.0,
            actor_kl=0.0,
            actor_step_l2=0.0,
        )

        def objective_and_kl() -> tuple[torch.Tensor, torch.Tensor]:
            current = self._actor.distribution(states)
            objective = (
                (current.log_prob(actions).sum(-1) - old_log_prob).exp() * advantages
            ).mean()
            kl = gaussian_kl(old_mean, old_std, current.mean, current.stddev)
            return objective, kl

        try:
            objective, _ = objective_and_kl()
            gradient = torch.cat(
                [g.reshape(-1) for g in torch.autograd.grad(objective, parameters)]
            ).detach()
            baseline = float(objective.detach())
            if not torch.isfinite(gradient).all() or not math.isfinite(baseline):
                reason, metrics["actor_numerical_failure"] = "nonfinite_gradient", 1.0
            elif gradient.norm().item() == 0:
                reason = "zero_gradient"
            else:

                def operator(vector: torch.Tensor) -> torch.Tensor:
                    return self.fisher_vector_product(states, old_mean, old_std, vector)

                cg = conjugate_gradient(
                    operator, gradient, self._cg_iterations, self._cg_tolerance
                )
                metrics.update(
                    cg_iterations=float(cg.iterations),
                    cg_residual_norm=cg.residual_norm,
                    cg_converged=float(cg.converged),
                    cg_breakdown=float(cg.breakdown),
                )
                curvature = cg.solution.dot(operator(cg.solution))
                if cg.breakdown or not torch.isfinite(curvature) or curvature <= 0:
                    reason, metrics["actor_numerical_failure"] = "cg_breakdown", 1.0
                else:
                    step = cg.solution * torch.sqrt(2 * self._target_kl / curvature)
                    expected = float(gradient.dot(step))
                    if (
                        not math.isfinite(expected)
                        or expected <= 0
                        or not torch.isfinite(step).all()
                    ):
                        reason, metrics["actor_numerical_failure"] = (
                            "invalid_direction",
                            1.0,
                        )
                    else:
                        for index in range(self._line_search_iterations):
                            fraction = self._line_search_shrink**index
                            self._set_actor_parameters(original + fraction * step)
                            metrics["line_search_backtracks"] = float(index)
                            try:
                                with torch.no_grad():
                                    candidate, kl = objective_and_kl()
                                finite = bool(
                                    torch.isfinite(candidate) & torch.isfinite(kl)
                                )
                            except (ValueError, RuntimeError):
                                finite = False
                            if not finite:
                                metrics["line_search_nonfinite_candidates"] += 1
                                continue
                            improvement = float(candidate) - baseline
                            if self._accept_actor_candidate(
                                improvement, float(kl), expected * fraction
                            ):
                                accepted, reason = True, None
                                metrics.update(
                                    surrogate_improvement=improvement,
                                    expected_improvement=expected * fraction,
                                )
                                break
                        if not accepted and metrics["line_search_nonfinite_candidates"]:
                            reason, metrics["actor_numerical_failure"] = (
                                "nonfinite_line_search",
                                1.0,
                            )
        finally:
            if not accepted:
                self._set_actor_parameters(original)
        # Measure the policy left in the actor, including after rollback.
        try:
            with torch.no_grad():
                final_objective, final_kl = objective_and_kl()
            if accepted and (
                not torch.isfinite(final_kl)
                or not torch.isfinite(final_objective)
                or float(final_kl) > self._target_kl
                or float(final_objective) <= baseline
            ):
                raise RuntimeError(
                    "accepted TRPO actor failed its final trust-region check"
                )
        except Exception:
            self._set_actor_parameters(original)
            raise
        metrics.update(
            actor_step_accepted=float(accepted),
            actor_step_rejected=float(not accepted),
            actor_kl=float(final_kl),
            actor_step_l2=float(
                (
                    torch.cat([p.detach().reshape(-1) for p in parameters]) - original
                ).norm()
            ),
        )
        self._last_actor_rejection_reason = reason
        return metrics

    def learn_rollout(
        self, batch: ContinuousPPOTransitionBatch
    ) -> dict[str, list[float]]:
        """Take one actor step, then fit values on the complete fresh rollout."""
        self._validate_rollout(batch)
        states = batch.state.detach()
        advantages = batch.advantages.detach()
        if self._normalize_advantages:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )
        metrics = self._actor_step(
            states,
            batch.latent_action.detach(),
            batch.old_log_prob.detach(),
            advantages,
        )
        self._actor_attempts += 1
        self._actor_accepted += int(metrics["actor_step_accepted"])
        self._actor_rejected += int(metrics["actor_step_rejected"])
        self._actor_numerical_failures += int(metrics["actor_numerical_failure"])
        # Consuming a rollout invalidates its behavior data even on rejection.
        self._policy_version += 1
        self._pending_action = None
        self._clear_behavior_sampling_metadata()
        report = {name: [value] for name, value in metrics.items()}
        report["actor_update_attempted"] = [1.0]
        report["value_loss"] = []
        for _ in range(self._critic_epochs):
            order = torch.randperm(len(batch), device=states.device)
            for indices in order.split(self._batch_size):
                loss = (
                    (
                        self._critic(states[indices]).squeeze(-1)
                        - batch.returns.detach()[indices]
                    )
                    .square()
                    .mean()
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite TRPO value loss")
                self._critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self._critic.parameters()
                ):
                    raise ValueError("nonfinite TRPO value gradient")
                if self._critic_max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._critic.parameters(),
                        self._critic_max_grad_norm,
                        error_if_nonfinite=True,
                    )
                self._critic_optimizer.step()
                self._critic_updates += 1
                report["value_loss"].append(float(loss.detach()))
        self._training_steps += 1
        return report

    def learn_batch(self, batch: ContinuousPPOTransitionBatch) -> dict[str, float]:
        return {
            name: sum(values) / len(values)
            for name, values in self.learn_rollout(batch).items()
        }

    def learn(self, replay_buffer: ReplayBuffer) -> dict[str, list[float]]:
        if (
            not isinstance(replay_buffer, ContinuousPPORolloutBuffer)
            or replay_buffer.policy is not self
        ):
            raise TypeError("use this policy's continuous on-policy rollout buffer")
        return (
            {}
            if not len(replay_buffer)
            else self.learn_rollout(replay_buffer.full_batch())
        )

    @property
    def actor_update_statistics(self) -> dict[str, Any]:
        return dict(
            attempted=self._actor_attempts,
            accepted=self._actor_accepted,
            rejected=self._actor_rejected,
            numerical_failures=self._actor_numerical_failures,
            last_rejection_reason=self._last_actor_rejection_reason,
            critic_optimizer_steps=self._critic_updates,
        )

    def get_extra_state(self) -> dict[str, Any]:
        return dict(
            version=1,
            config=copy.deepcopy(self._checkpoint_config),
            critic_optimizer=copy.deepcopy(self._critic_optimizer.state_dict()),
            policy_version=self._policy_version,
            training_steps=self._training_steps,
            counters=self.actor_update_statistics,
        )

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1 or state.get("config") != self._checkpoint_config:
            raise ValueError("TRPO checkpoint configuration mismatch")
        self._critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._policy_version, self._training_steps = int(state["policy_version"]), int(
            state["training_steps"]
        )
        counters = state["counters"]
        self._actor_attempts, self._actor_accepted = (
            counters["attempted"],
            counters["accepted"],
        )
        self._actor_rejected, self._actor_numerical_failures = (
            counters["rejected"],
            counters["numerical_failures"],
        )
        self._critic_updates = counters["critic_optimizer_steps"]
        self._last_actor_rejection_reason = counters["last_rejection_reason"]
        self._pending_action = None
        self._clear_behavior_sampling_metadata()

    def _apply(
        self, function: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> "ContinuousTrustRegionPolicyOptimization":
        result = super()._apply(function, recurse)
        optimizer = getattr(self, "_critic_optimizer", None)
        if optimizer is not None:
            for state in optimizer.state.values():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[name] = function(value)
        return result

    def compare(self, other: PolicyLearner) -> str:
        if type(other) is not type(self):
            return "TRPO learner types differ"
        differences = [PolicyLearner.compare(self, other)]
        if self._checkpoint_config != other._checkpoint_config:
            differences.append("TRPO configuration differs")
        if (
            self.actor_update_statistics != other.actor_update_statistics
            or self._policy_version != other._policy_version
            or self._training_steps != other._training_steps
        ):
            differences.append("TRPO update counters differ")
        optimizer_states = [
            copy.deepcopy(policy._critic_optimizer.state_dict())
            for policy in (self, other)
        ]
        for state in optimizer_states:
            for values in state["state"].values():
                for name, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[name] = value.cpu()
        if reason := state_dict_owners_have_similar_state_dict(*optimizer_states):
            differences.append("TRPO critic optimizer differs: " + reason)
        for name in ["_actor", "_critic"]:
            left, right = (
                getattr(self, name).state_dict(),
                getattr(other, name).state_dict(),
            )
            if any(not torch.equal(left[key].cpu(), right[key].cpu()) for key in left):
                differences.append(name + " parameters differ")
        return "\n".join(item for item in differences if item)
