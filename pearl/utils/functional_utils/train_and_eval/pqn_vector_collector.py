# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Collect native Pearl environments with batched policy inference.

Environments step synchronously on CPU. The learner and rollout tensors
can use CUDA.
"""

from collections.abc import Callable
from typing import Any

import torch
from pearl.api.environment import Environment
from pearl.replay_buffers.sequential_decision_making.pqn_rollout_buffer import (
    PQNRollout,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class PQNVectorCollector:
    """Collect independent environment streams for a batched learner.

    The learner supplies act_batch(states, unavailable_actions_mask, epsilon)
    and PyTorch parameters. Each environment resets independently. Its final
    observation is saved before reset, so returns respect episode boundaries.
    Rewards are unchanged. Environments reporting costs are rejected.
    Identity histories are required.
    """

    def __init__(
        self,
        environments: list[Environment],
        learner: Any,
        seed: int,
        num_actions: int | None = None,
    ) -> None:
        if not environments:
            raise ValueError("at least one environment is required")
        self.environments, self.learner, self.seed = list(environments), learner, seed
        self.device = next(learner.parameters()).device
        if num_actions is None:
            num_actions = learner.action_representation_module.max_number_actions
        if (
            isinstance(num_actions, bool)
            or not isinstance(num_actions, int)
            or num_actions < 1
        ):
            raise ValueError("fixed catalog size must be a positive integer")
        self.num_actions = num_actions
        self.steps = 0
        self.episodes: list[dict[str, Any]] = []
        self._episode_ids = [0] * len(environments)
        self._returns = [0.0] * len(environments)
        self._lengths = [0] * len(environments)
        self._observations: list[torch.Tensor] = []
        self._action_spaces: list[DiscreteActionSpace] = []
        for index in range(len(environments)):
            observation, actions = self._reset(index)
            self._observations.append(observation)
            self._action_spaces.append(actions)
        if len({tuple(x.shape) for x in self._observations}) != 1:
            raise ValueError("all environment observation shapes must match")

    def _mask(self, space: DiscreteActionSpace) -> torch.Tensor:
        if not isinstance(space, DiscreteActionSpace) or space.action_dim != 1:
            raise ValueError("vector collection requires scalar discrete catalog IDs")
        ids = space.actions_batch.detach().cpu().reshape(-1)
        if (
            ids.numel() == 0
            or ids.dtype == torch.bool
            or ids.is_complex()
            or not torch.isfinite(ids).all()
            or not torch.equal(ids, ids.round())
            or (ids < 0).any()
            or (ids >= self.num_actions).any()
            or ids.unique().numel() != ids.numel()
        ):
            raise ValueError("available actions must be unique IDs within the catalog")
        mask = torch.ones(self.num_actions, dtype=torch.bool)
        mask[ids.long()] = False
        return mask

    def _reset(self, index: int) -> tuple[torch.Tensor, DiscreteActionSpace]:
        seed = (self.seed + 97_409 * index + 1_000_003 * self._episode_ids[index]) % (
            2**31 - 1
        )
        observation, actions = self.environments[index].reset(seed=seed)
        self._mask(actions)
        observation = (
            torch.as_tensor(observation, dtype=torch.float32)
            .detach()
            .cpu()
            .reshape(-1)
            .clone()
        )
        if not torch.isfinite(observation).all():
            raise ValueError("reset observations must be finite")
        return observation, actions

    def collect(
        self, num_steps: int, epsilon: float | Callable[[int], float] | None = None
    ) -> PQNRollout:
        """Collect ``num_steps * num_environments`` actual transitions.

        An epsilon callable receives completed actual transitions before each
        vector step. Its schedule is never measured in vector calls by accident.
        """
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps < 1
        ):
            raise ValueError("num_steps must be a positive integer")
        rows: list[dict[str, torch.Tensor]] = []
        for _ in range(num_steps):
            states = torch.stack(self._observations)
            masks = torch.stack([self._mask(s) for s in self._action_spaces])
            value = epsilon(self.steps) if callable(epsilon) else epsilon
            with torch.no_grad():
                actions = (
                    self.learner.act_batch(
                        states.to(self.device), masks.to(self.device), epsilon=value
                    )
                    .detach()
                    .cpu()
                )
            if actions.shape != (len(self.environments), 1):
                raise ValueError(
                    "batched actor must return [environments, 1] action IDs"
                )
            next_states, rewards, terminated, truncated, next_masks = [], [], [], [], []
            for index, env in enumerate(self.environments):
                action = actions[index]
                if (
                    action.dtype == torch.bool
                    or action.is_complex()
                    or not torch.isfinite(action).all()
                    or not torch.equal(action, action.round())
                    or int(action.item()) < 0
                    or int(action.item()) >= self.num_actions
                    or masks[index, int(action.item())]
                ):
                    raise ValueError("actor selected an invalid or unavailable action")
                result = env.step(action)
                if result.cost is not None:
                    raise ValueError(
                        "PQN collector is reward-only; cost handling must be explicit upstream"
                    )
                final_observation = result.observation
                if result.done:
                    for key in (
                        "final_observation",
                        "final_obs",
                        "terminal_observation",
                    ):
                        if (result.info or {}).get(key) is not None:
                            final_observation = result.info[key]
                            break
                if final_observation is None:
                    if not result.terminated:
                        raise ValueError(
                            "a truncated transition requires its final observation"
                        )
                    final_state = torch.zeros_like(states[index])
                else:
                    final_state = (
                        torch.as_tensor(final_observation, dtype=torch.float32)
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .clone()
                    )
                if final_state.shape != states[index].shape:
                    raise ValueError("transition observation shape changed")
                final_actions = (
                    self._action_spaces[index]
                    if result.available_action_space is None
                    else result.available_action_space
                )
                next_states.append(final_state)
                next_masks.append(self._mask(final_actions))
                rewards.append(float(result.reward))
                terminated.append(bool(result.terminated))
                truncated.append(bool(result.truncated))
                self._returns[index] += float(result.reward)
                self._lengths[index] += 1
                if result.done:
                    self.episodes.append(
                        dict(
                            environment_index=index,
                            episode=self._episode_ids[index],
                            return_=self._returns[index],
                            length=self._lengths[index],
                            end_transition=self.steps + index + 1,
                            terminated=bool(result.terminated),
                            truncated=bool(result.truncated),
                        )
                    )
                    self._episode_ids[index] += 1
                    self._returns[index], self._lengths[index] = 0.0, 0
                    self._observations[index], self._action_spaces[index] = self._reset(
                        index
                    )
                else:
                    self._observations[index], self._action_spaces[index] = (
                        final_state,
                        final_actions,
                    )
            rows.append(
                dict(
                    states=states,
                    actions=actions.clone(),
                    rewards=torch.tensor(rewards),
                    terminated=torch.tensor(terminated, dtype=torch.bool),
                    truncated=torch.tensor(truncated, dtype=torch.bool),
                    next_states=torch.stack(next_states),
                    next_unavailable_actions_mask=torch.stack(next_masks),
                )
            )
            self.steps += len(self.environments)
        rollout = PQNRollout(
            **{key: torch.stack([r[key] for r in rows]) for key in rows[0]}
        )
        return rollout.to(self.device)

    def snapshot(self) -> dict[str, Any]:
        """Diagnostic state only; this cannot restore simulator or RNG state."""
        return dict(
            steps=self.steps,
            num_environments=len(self.environments),
            seed=self.seed,
            observations=[x.tolist() for x in self._observations],
            episode_ids=list(self._episode_ids),
            incomplete_episode_returns=list(self._returns),
            incomplete_episode_lengths=list(self._lengths),
            completed_episodes=list(self.episodes),
            backend="synchronous native CPU environments; batched policy inference",
        )

    def close(self) -> None:
        for env in self.environments:
            env.close()
