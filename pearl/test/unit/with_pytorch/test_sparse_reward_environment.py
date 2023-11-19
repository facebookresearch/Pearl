#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.utils.instantiations.environments.sparse_reward_environment import (
    DiscreteSparseRewardEnvironment,
)


class TestSparseRewardEnvironment(unittest.TestCase):
    def test_basic(self) -> None:
        env = DiscreteSparseRewardEnvironment(
            length=100, height=100, step_size=1, action_count=4
        )

        # Test reset
        observation, action_space = env.reset()
        self.assertEqual(4, action_space.n)
        x, y = observation.agent_position
        self.assertLess(x, 100)
        self.assertLess(y, 100)
        goal_x, goal_y = observation.goal
        self.assertLess(goal_x, 100)
        self.assertLess(goal_y, 100)

        # Test position change
        result = env.step(torch.tensor(0))
        self.assertEqual(result.observation.agent_position[0], x + 1)
        self.assertEqual(result.observation.agent_position[1], y)
        x = x + 1
        result = env.step(torch.tensor(1))
        self.assertEqual(result.observation.agent_position[0], x)
        self.assertEqual(result.observation.agent_position[1], y + 1)
        y = y + 1
        result = env.step(torch.tensor(2))
        self.assertEqual(result.observation.agent_position[0], x - 1)
        self.assertEqual(result.observation.agent_position[1], y)
        x = x - 1
        result = env.step(torch.tensor(3))
        self.assertEqual(result.observation.agent_position[0], x)
        self.assertEqual(result.observation.agent_position[1], y - 1)

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
        self.assertTrue(result.terminated)
