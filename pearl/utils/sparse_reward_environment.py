"""
This file contains environment to simulate sparse rewards

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
from collections import namedtuple

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace

from pearl.api.environment import Environment
from pearl.utils.action_spaces import DiscreteActionSpace

SparseRewardEnvironmentObservation = namedtuple(
    "SparseRewardEnvironmentObservation", ["agent_position", "goal"]
)


class SparseRewardEnvironment(Environment):
    def __init__(
        self,
        length: float,
        height: float,
        max_episode_duration: int = 500,
        reward_distance: float = 1,
    ):
        self._length = length
        self._height = height
        self._max_episode_duration = max_episode_duration
        # reset will initialize following
        self._agent_position = None
        self._goal = None
        self._step_count = 0
        self._reward_distance = reward_distance

    @abstractmethod
    def step(self, action: Action) -> ActionResult:
        pass

    def reset(self) -> (SparseRewardEnvironmentObservation, ActionSpace):
        # reset (x, y)
        self._agent_position = (self._length / 2, self._height / 2)
        self._goal = (random.uniform(0, self._length), random.uniform(0, self._height))
        self._step_count = 0
        return (
            SparseRewardEnvironmentObservation(
                agent_position=self._agent_position, goal=self._goal
            ),
            None,
        )

    def _update_position(self, delta) -> None:
        """
        This API is to update and clip and ensure agent always stay in map
        """
        delta_x, delta_y = delta
        x, y = self._agent_position
        self._agent_position = (
            max(min(x + delta_x, self._length), 0),
            max(min(y + delta_y, self._height), 0),
        )


class DiscreteSparseRewardEnvironment(SparseRewardEnvironment):
    """
    Given action count N, action index will be 0,...,N-1
    For action n, position will be changed by:
    x +=  cos(360/N * n) * step_size
    y +=  sin(360/N * n) * step_size
    """

    def __init__(
        self,
        length: float,
        height: float,
        step_size: float = 0.01,
        action_count: int = 4,
        max_episode_duration: int = 500,
        reward_distance: float = None,
    ):
        super(DiscreteSparseRewardEnvironment, self).__init__(
            length,
            height,
            max_episode_duration,
            reward_distance if reward_distance is not None else step_size,
        )
        self._step_size = step_size
        self._action_count = action_count
        self._actions = [
            (
                math.cos(2 * math.pi / self._action_count * i),
                math.sin(2 * math.pi / self._action_count * i),
            )
            * self._step_size
            for i in range(action_count)
        ]

    def _check_win(self) -> bool:
        """
        Return:
            True if reached goal
            False if not reached goal
        """
        if math.dist(self._agent_position, self._goal) < self._reward_distance:
            return True
        return False

    def step(self, action: Action) -> ActionResult:
        assert action < self._action_count and action >= 0
        # update position
        self._update_position(self._actions[action])

        has_win = self._check_win()
        self._step_count += 1
        terminated = has_win or self._step_count >= self._max_episode_duration
        return ActionResult(
            observation=SparseRewardEnvironmentObservation(
                agent_position=self._agent_position, goal=self._goal
            ),
            reward=0 if has_win else -1,
            terminated=terminated,
            truncated=False,
            info=None,
        )

    def reset(self) -> (SparseRewardEnvironmentObservation, ActionSpace):
        observation, _ = super(DiscreteSparseRewardEnvironment, self).reset()
        return observation, self.action_space

    @property
    def action_space(self) -> DiscreteActionSpace:
        return DiscreteActionSpace(range(self._action_count))
