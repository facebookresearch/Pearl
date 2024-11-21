# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch

from pearl.utils.instantiations.environments.sparse_reward_environment import (
    DiscreteSparseRewardEnvironment,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestSparseRewardEnvironment(unittest.TestCase):
    def test_basic(self) -> None:
        env = DiscreteSparseRewardEnvironment(
            width=100, height=100, reward_distance=1, step_size=1, action_count=4
        )

        # Test reset
        observation, action_space = env.reset()
        assert isinstance(action_space, DiscreteActionSpace)
        self.assertEqual(4, action_space.n)
        # FIXME: private attributes should not be accessed.
        assert env._agent_position is not None
        x, y = env._agent_position
        self.assertLess(x, 100)
        self.assertLess(y, 100)
        assert env._goal is not None
        goal_x, goal_y = env._goal
        self.assertLess(goal_x, 100)
        self.assertLess(goal_y, 100)

        # Test position change
        result = env.step(torch.tensor(0))
        assert (agent_position := env._agent_position) is not None
        self.assertEqual(agent_position[0], x + 1)
        self.assertEqual(agent_position[1], y)
        x = x + 1

        result = env.step(torch.tensor(1))
        assert (agent_position := env._agent_position) is not None
        self.assertEqual(agent_position[0], x)
        self.assertEqual(agent_position[1], y + 1)
        y = y + 1

        result = env.step(torch.tensor(2))
        assert (agent_position := env._agent_position) is not None
        self.assertEqual(agent_position[0], x - 1)
        self.assertEqual(agent_position[1], y)
        x = x - 1

        result = env.step(torch.tensor(3))
        assert (agent_position := env._agent_position) is not None
        self.assertEqual(agent_position[0], x)
        self.assertEqual(agent_position[1], y - 1)

        # Test win reward and terminate
        env._agent_position = (goal_x, goal_y - 1)
        result = env.step(torch.tensor(1))
        self.assertEqual(result.reward, 0)
        self.assertTrue(result.terminated)
        # Test not win reward and not terminate
        env._agent_position = (goal_x - 10, goal_y - 10)
        result = env.step(torch.tensor(1))
        self.assertEqual(result.reward, -1)
        self.assertFalse(result.terminated)
        # Test not win reward and terminate
        env._agent_position = (goal_x - 10, goal_y - 10)
        env._step_count = env._max_episode_duration
        result = env.step(torch.tensor(1))
        self.assertEqual(result.reward, -1)
        self.assertTrue(result.truncated)
