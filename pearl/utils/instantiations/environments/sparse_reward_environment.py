# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

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
- one for contineous action space
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
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class SparseRewardEnvironment(Environment):
    def __init__(
        self,
        length: float,
        height: float,
        max_episode_duration: int = 500,
        reward_distance: float = 1,
    ) -> None:
        self._length = length
        self._height = height
        self._max_episode_duration = max_episode_duration
        # reset will initialize following
        self._agent_position: Optional[Tuple[float, float]] = None
        self._goal: Optional[Tuple[float, float]] = None
        self._step_count = 0
        self._reward_distance = reward_distance

    @abstractmethod
    def step(self, action: Action) -> ActionResult:
        pass

    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, ActionSpace]:

        # reset (x, y)
        self._agent_position = (self._length / 2, self._height / 2)
        self._goal = (random.uniform(0, self._length), random.uniform(0, self._height))
        self._step_count = 0
        assert self._agent_position is not None
        assert (goal := self._goal) is not None
        return (
            torch.tensor(list(self._agent_position) + list(goal)),
            self.action_space,
        )

    def _update_position(self, delta: Tuple[float, float]) -> None:
        """
        This API is to update and clip and ensure agent always stay in map
        """
        delta_x, delta_y = delta
        assert self._agent_position is not None
        x, y = self._agent_position
        self._agent_position = (
            max(min(x + delta_x, self._length), 0),
            max(min(y + delta_y, self._height), 0),
        )

    def _check_win(self) -> bool:
        """
        Return:
            True if reached goal
            False if not reached goal
        """
        assert self._agent_position is not None
        assert self._goal is not None
        if math.dist(self._agent_position, self._goal) < self._reward_distance:
            return True
        return False


class ContinuousSparseRewardEnvironment(SparseRewardEnvironment):
    """
    Given action vector (x, y)
    agent position is updated accordingly
    """

    def step(self, action: Action) -> ActionResult:
        assert isinstance(action, torch.Tensor)
        self._update_position((action[0].item(), action[1].item()))

        has_win = self._check_win()
        self._step_count += 1
        terminated = has_win or self._step_count >= self._max_episode_duration
        assert self._agent_position is not None
        assert (goal := self._goal) is not None
        return ActionResult(
            observation=torch.tensor(list(self._agent_position) + list(goal)),
            reward=0 if has_win else -1,
            terminated=terminated,
            truncated=False,
        )

    @property
    def action_space(self) -> ActionSpace:
        # pyre-fixme[7]: Expected `ActionSpace` but got `None`.
        # FIXME: does this really do not have an action space?
        return None


class DiscreteSparseRewardEnvironment(ContinuousSparseRewardEnvironment):
    """
    Given action count N, action index will be 0,...,N-1
    For action n, position will be changed by:
    x +=  cos(360/N * n) * step_size
    y +=  sin(360/N * n) * step_size
    """

    # FIXME: This environment mixes the concepts of action index and action feature.
    def __init__(
        self,
        length: float,
        height: float,
        step_size: float = 0.01,
        action_count: int = 4,
        max_episode_duration: int = 500,
        reward_distance: Optional[float] = None,
    ) -> None:
        super(DiscreteSparseRewardEnvironment, self).__init__(
            length,
            height,
            max_episode_duration,
            reward_distance if reward_distance is not None else step_size,
        )
        self._step_size = step_size
        self._action_count = action_count
        self._actions: List[torch.Tensor] = [
            torch.tensor(
                [
                    math.cos(2 * math.pi / self._action_count * i),
                    math.sin(2 * math.pi / self._action_count * i),
                ]
            )
            * self._step_size
            for i in range(action_count)
        ]

    def step(self, action: Action) -> ActionResult:
        assert action < self._action_count and action >= 0
        return super(DiscreteSparseRewardEnvironment, self).step(
            self._actions[int(action.item())]
        )

    @property
    def action_space(self) -> DiscreteActionSpace:
        return DiscreteActionSpace(
            actions=[torch.tensor([i]) for i in range(self._action_count)]
        )
