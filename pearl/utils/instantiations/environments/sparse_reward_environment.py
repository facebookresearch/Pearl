# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

"""
This file contains environment to simulate sparse rewards
Also contains history summarization module that needs to be used together
when defining PearlAgent

Set up is following:
2d box environment, where the agent gets initialized in a center of a square arena,
and there is a target - 2d point, randomly generated in the arena.
The agent gets reward 0 only when it gets close enough to the target, otherwise the reward is -1.

There are 2 versions in this file:
- one for discrete action space
- one for continuous action space
"""

import math
import random
from abc import abstractmethod
from typing import List, Optional, Tuple

import torch

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace

from pearl.api.environment import Environment
from pearl.utils.instantiations.spaces.box import BoxSpace
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace



class SparseRewardEnvironment(Environment):
    def __init__(
        self,
        width: float,
        height: float,
        max_episode_duration: int = 500,
        reward_distance: float = 1.0,
    ) -> None:
        """Initialize a sparse-reward environment.

        Args:
            width: Arena width (x-dimension).
            height: Arena height (y-dimension).
            max_episode_duration: Step limit per episode.
            reward_distance: Distance threshold for reaching the goal.
        """
        self._width = width
        self._height = height
        self._max_episode_duration = max_episode_duration
        self._reward_distance = reward_distance

        # reset will initialize the agent position, goal and step count
        self._agent_position: tuple[float, float] | None = None
        self._goal: tuple[float, float] | None = None
        self._step_count = 0

    @abstractmethod
    def step(self, action: Action) -> ActionResult:
        """Take one step in the environment"""
        raise NotImplementedError

    @property
    def observation_space(self) -> BoxSpace:
        """
        Observations consist of the agent position ``(x, y)`` concatenated with
        the goal position ``(goal_x, goal_y)``.  Each coordinate is bounded in
        the interval ``[0, width]`` for ``x`` and ``[0, height]`` for ``y``.
        """
        low = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        high = torch.tensor([self._length, self._height, self._length, self._height], dtype=torch.float32)
        return BoxSpace(low=low, high=high)

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, ActionSpace]:
        """Resets the environment and returns the initial observation and initial action space."""
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        # reset (x, y) for agent position
        self._agent_position = (
            self._width / 2.0,
            self._height / 2.0,
        )

        # reset (x, y) for goal
        self._goal = (
            random.uniform(0, self._width),
            random.uniform(0, self._height),
        )

        self._step_count = 0  # reset step_count
        assert self._agent_position is not None
        assert (goal := self._goal) is not None
        return (
            torch.tensor(
                list(self._agent_position) + list(goal), dtype=torch.float32
            ),
            self.action_space,
        )

    def _update_position(self, delta: tuple[float, float]) -> None:
        """
        Update the agent position, say (x, y) --> (x', y') where:
        x' = x + delta_x
        y' = y + delta_y

        A clip operation is added to ensure the agent always stay in 2d grid.
        """
        delta_x, delta_y = delta
        assert self._agent_position is not None
        x, y = self._agent_position
        self._agent_position = (
            max(min(x + delta_x, self._width), 0.0),
            max(min(y + delta_y, self._height), 0.0),
        )

    def _check_win(self) -> bool:
        """
        Indicates whether the agent position is close enough (in Euclidean distance) to the goal.
        """
        assert self._agent_position is not None
        assert self._goal is not None

        if math.dist(self._agent_position, self._goal) < self._reward_distance:
            return True
        return False


class ContinuousSparseRewardEnvironment(SparseRewardEnvironment):
    """
    Action is a 2D tensor (dx, dy) indicating movement in the x and y directions.
    The agent's position is updated by this delta each step.
    """

    def step(self, action: Action) -> ActionResult:
        """
        Perform one environment step.

        Args:
            action : torch.Tensor | numpy.ndarray
                A 2-D displacement (dx, dy).  Accepts anything convertible to a
                length-2 float tensor.

        Returns:
            ActionResult
                Observation (4-D), reward (float), and termination flags.
        """
        assert isinstance(action, torch.Tensor), "Action must be a torch.Tensor for continuous environment"
        assert action.shape == (2,), f"Continuous action should be shape (2,), got {action.shape}"
        dx, dy = float(action[0].item()), float(action[1].item())
        self._update_position((dx, dy))

        has_win = self._check_win()
        self._step_count += 1
        terminated = has_win
        truncated = (not has_win) and self._step_count >= self._max_episode_duration
        assert self._agent_position is not None
        assert (goal := self._goal) is not None
        return ActionResult(
            observation=torch.tensor(
                list(self._agent_position) + list(goal), dtype=torch.float32
            ),
            reward=0.0 if has_win else -1.0,
            terminated=terminated,
            truncated=truncated,
        )

    @property
    def action_space(self) -> ActionSpace:
        """
        Continuous 2-D `BoxActionSpace` for (dx, dy).

        Boundaries are set to [-1, 1] in each direction; agents are free
        to scale internally if they prefer a different magnitude.
        """
        low = torch.tensor([-1.0, -1.0], dtype=torch.float32)
        high = torch.tensor([1.0, 1.0], dtype=torch.float32)
        return BoxActionSpace(low=low, high=high)


class DiscreteSparseRewardEnvironment(ContinuousSparseRewardEnvironment):
    """
    Action space has `action_count` discrete moves uniformly distributed in 360 degrees.
    For action index n (0 <= n < N), the agent moves by `step_size` in the direction of angle (2pi * n / N).
    """
    def __init__(
        self,
        width: float,
        height: float,
        action_count: int = 4,
        reward_distance: float | None = None,
        step_size: float = 0.01,
        max_episode_duration: int = 500,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            max_episode_duration=max_episode_duration,
            # If no specific reward_distance is given, use step_size as the threshold (so one step reach counts as success)
            reward_distance=reward_distance if reward_distance is not None else step_size,
        )
        if action_count <= 0:
            raise ValueError("action_count must be a positive integer")
        self._step_size = step_size
        self._action_count = action_count
        # Pre-compute discrete action deltas (as continuous vectors) for each action index
        self._actions: List[torch.Tensor] = []
        for i in range(action_count):
            angle = 2.0 * math.pi * i / action_count
            dx = math.cos(angle) * self._step_size
            dy = math.sin(angle) * self._step_size
            # Each action vector is a 2D tensor (dx, dy)
            self._actions.append(torch.tensor([dx, dy], dtype=torch.float32))

    def step(self, action: Action) -> ActionResult:
        """
        Performs a single discrete action.

        Args:
            action : int | torch.Tensor
                Either a Python int or a 0-D / 1-D tensor holding the index.

        Returns:
            ActionResult
                Observation, reward, and termination flags after the move.
        """
        if isinstance(action, torch.Tensor):
            if action.ndim > 1:
                raise ValueError(
                    f"Discrete action tensor must be scalar or 1-D of length 1, got shape {list(action.shape)}"
                )
            idx = int(action.item())
        else:
            idx = int(action)
        if not (0 <= idx < self._action_count):
            raise ValueError(
                f"Action index {idx} out of range; must be 0 <= idx < {self._action_count}"
            )
        continuous_action = self._actions[ind]
        return super().step(continuous_action)

    @property
    def action_space(self) -> DiscreteActionSpace:
        """
        Returns a `DiscreteActionSpace` containing `action_count` actions.
        Each action is a scalar tensor holding its own index (0, 1, ... N-1).
        """
        return DiscreteActionSpace(
            actions=[torch.tensor([i], dtype=torch.int) for i in range(self._action_count)]
        )
