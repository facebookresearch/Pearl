# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Native environments with batched continuous PPO/RPO behavior sampling.

Physics runs in synchronous CPU environments; policy inference/optimization can
run on CUDA. Return/observation normalization is explicit and shared across the
environment streams, with frozen statistics for evaluation. Time-limit final
observations are preserved before per-environment resets.
"""

import math
from typing import Any

import torch
from pearl.api.environment import Environment
from pearl.policy_learners.sequential_decision_making.continuous_ppo import (
    ContinuousProximalPolicyOptimization,
    ContinuousPPOTransitionBatch,
    generalized_advantage_estimate,
)


class RunningMoments:
    """Parallel-merge population moments, retained in CPU float64."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.variance = torch.ones(shape, dtype=torch.float64)
        self.count = 1e-4

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu", dtype=torch.float64)
        if (
            values.ndim != self.mean.ndim + 1
            or values.shape[1:] != self.mean.shape
            or values.shape[0] < 1
            or not torch.isfinite(values).all()
        ):
            raise ValueError("nonempty finite observations required")
        count = values.shape[0]
        mean = values.mean(0)
        variance = values.var(0, unbiased=False)
        delta = mean - self.mean
        total = self.count + count
        self.variance = (
            self.variance * self.count
            + variance * count
            + delta.square() * self.count * count / total
        ) / total
        self.mean = self.mean + delta * count / total
        self.count = total

    def snapshot(self) -> dict[str, Any]:
        return dict(
            mean=self.mean.tolist(), variance=self.variance.tolist(), count=self.count
        )


def normalize_from_snapshot(
    states: torch.Tensor, snapshot: dict[str, Any]
) -> torch.Tensor:
    """Frozen evaluation preprocessing; simulator/RNG state is not restored."""
    states = torch.as_tensor(states, dtype=torch.float32, device="cpu")
    if not snapshot["normalize_observations"]:
        return states
    moments = snapshot["observation_moments"]
    mean = torch.as_tensor(moments["mean"], dtype=torch.float64)
    variance = torch.as_tensor(moments["variance"], dtype=torch.float64)
    if (
        mean.ndim != 1
        or mean.shape != variance.shape
        or states.shape[-1] != mean.numel()
        or not torch.isfinite(mean).all()
        or not torch.isfinite(variance).all()
        or (variance < 0).any()
        or not math.isfinite(moments["count"])
        or moments["count"] <= 0
    ):
        raise ValueError("invalid normalization snapshot")
    return ((states - mean) / (variance + 1e-8).sqrt()).clamp(-10, 10).float()


class ContinuousPPOVectorCollector:
    """Fresh [time, environment] rollouts, flattened only after GAE computation."""

    def __init__(
        self,
        environments: list[Environment],
        policy: ContinuousProximalPolicyOptimization,
        seed: int,
        normalize_observations: bool = True,
        normalize_rewards: bool = True,
    ) -> None:
        if not environments:
            raise ValueError("at least one environment required")
        self.environments, self.policy, self.seed = list(environments), policy, seed
        self.device = next(policy.parameters()).device
        self.normalize_observations, self.normalize_rewards = (
            normalize_observations,
            normalize_rewards,
        )
        self.steps = 0
        self.episodes: list[dict[str, Any]] = []
        n = len(environments)
        self._episode_ids = [0] * n
        self._returns = [0.0] * n
        self._lengths = [0] * n
        self._discounted_returns = torch.zeros(n, dtype=torch.float64)
        raw = torch.stack([self._reset(i) for i in range(n)])
        self.observation_moments = RunningMoments((raw.shape[-1],))
        self.return_moments = RunningMoments(())
        if normalize_observations:
            self.observation_moments.update(raw)
        self._states = self.normalize(raw)

    def _reset(self, index: int) -> torch.Tensor:
        seed = (self.seed + 97_409 * index + 1_000_003 * self._episode_ids[index]) % (
            2**31 - 1
        )
        state, actions = self.environments[index].reset(seed=seed)
        self.policy._validate_action_space(actions)
        state = torch.as_tensor(state, dtype=torch.float32).cpu().reshape(-1).clone()
        if not torch.isfinite(state).all():
            raise ValueError("nonfinite reset observation")
        return state

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        states = torch.as_tensor(states, dtype=torch.float32, device="cpu")
        if not self.normalize_observations:
            return states
        return (
            (
                (states - self.observation_moments.mean)
                / (self.observation_moments.variance + 1e-8).sqrt()
            )
            .clamp(-10, 10)
            .float()
        )

    def collect(self, num_steps: int) -> ContinuousPPOTransitionBatch:
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps < 1
        ):
            raise ValueError("positive vector step count required")
        version = self.policy._policy_version
        rows = []
        n = len(self.environments)
        for _ in range(num_steps):
            states = self._states.clone()
            action, latent, log_prob, values = self.policy.sample_actions(
                states.to(self.device)
            )
            action_cpu = action.cpu()
            raw_next, reward, terminated, truncated = [], [], [], []
            resets = {}
            for index, env in enumerate(self.environments):
                result = env.step(action_cpu[index])
                if result.cost is not None:
                    raise ValueError("continuous PPO collector is reward-only")
                observation = result.observation
                if result.done:
                    for key in [
                        "final_observation",
                        "final_obs",
                        "terminal_observation",
                    ]:
                        if (result.info or {}).get(key) is not None:
                            observation = result.info[key]
                            break
                if observation is None:
                    if not result.terminated:
                        raise ValueError(
                            "truncation needs its actual final observation"
                        )
                    final = torch.zeros_like(states[index])
                else:
                    final = (
                        torch.as_tensor(observation, dtype=torch.float32)
                        .cpu()
                        .reshape(-1)
                        .clone()
                    )
                if (
                    final.shape != states[index].shape
                    or not torch.isfinite(final).all()
                ):
                    raise ValueError("nonfinite or changed final observation shape")
                if not math.isfinite(float(result.reward)):
                    raise ValueError("nonfinite reward")
                raw_next.append(final)
                reward.append(float(result.reward))
                terminated.append(bool(result.terminated))
                truncated.append(bool(result.truncated))
                self._returns[index] += float(result.reward)
                self._lengths[index] += 1
                if result.done:
                    self.episodes.append(
                        dict(
                            environment_index=index,
                            return_=self._returns[index],
                            length=self._lengths[index],
                            end_transition=self.steps + index + 1,
                        )
                    )
                    self._returns[index], self._lengths[index] = 0.0, 0
                    self._episode_ids[index] += 1
                    resets[index] = self._reset(index)
            finals = torch.stack(raw_next)
            if self.normalize_observations:
                self.observation_moments.update(finals)
                if resets:
                    self.observation_moments.update(torch.stack(list(resets.values())))
            normalized_finals = self.normalize(finals)
            next_states = finals.clone()
            for index, state in resets.items():
                next_states[index] = state
            self._states = self.normalize(next_states)
            term = torch.tensor(terminated, dtype=torch.bool)
            trunc = torch.tensor(truncated, dtype=torch.bool)
            with torch.no_grad():
                next_values = self.policy._critic(
                    normalized_finals.to(self.device)
                ).squeeze(-1)
                next_values = next_values.masked_fill(term.to(self.device), 0.0)
            rewards = torch.tensor(reward, dtype=torch.float32)
            self._discounted_returns = (
                self._discounted_returns * self.policy._discount_factor
                + rewards.double()
            )
            if self.normalize_rewards:
                self.return_moments.update(self._discounted_returns)
                rewards = (
                    (rewards / (self.return_moments.variance + 1e-8).sqrt())
                    .clamp(-10, 10)
                    .float()
                )
            self._discounted_returns[term | trunc] = 0.0
            rows.append(
                dict(
                    state=states,
                    action=action_cpu,
                    latent_action=latent.cpu(),
                    old_log_prob=log_prob.cpu(),
                    old_value=values.cpu(),
                    reward=rewards,
                    terminated=term,
                    truncated=trunc,
                    next_value=next_values.cpu(),
                )
            )
            self.steps += n
        tensors = {
            name: torch.stack([row[name] for row in rows]).to(self.device)
            for name in rows[0]
        }
        advantage, returns = generalized_advantage_estimate(
            tensors["reward"],
            tensors["old_value"],
            tensors["next_value"],
            tensors["terminated"],
            tensors["truncated"],
            self.policy._discount_factor,
            self.policy._trace_decay,
        )
        fields = {
            name: value.flatten(0, 1)
            for name, value in tensors.items()
            if name != "next_value"
        }
        if self.policy._policy_version != version:
            raise ValueError("policy changed during rollout collection")
        return ContinuousPPOTransitionBatch(
            **fields,
            advantages=advantage.flatten(),
            returns=returns.flatten(),
            behavior_policy_version=version,
        )

    def snapshot(self) -> dict[str, Any]:
        return dict(
            steps=self.steps,
            num_environments=len(self.environments),
            seed=self.seed,
            normalize_observations=self.normalize_observations,
            normalize_rewards=self.normalize_rewards,
            observation_moments=self.observation_moments.snapshot(),
            return_moments=self.return_moments.snapshot(),
            incomplete_returns=list(self._returns),
            incomplete_lengths=list(self._lengths),
            episodes=list(self.episodes),
        )

    def close(self) -> None:
        for env in self.environments:
            env.close()
