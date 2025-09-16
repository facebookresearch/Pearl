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
    ContinuousSparseRewardEnvironment,
    DiscreteSparseRewardEnvironment,
)
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestDiscreteSparseRewardEnvironment(unittest.TestCase):
    def test_discrete_actions(self) -> None:
        width, height = 100, 100
        env = DiscreteSparseRewardEnvironment(
            width=width, height=height, reward_distance=1, step_size=1, action_count=4
        )

        # Test reset
        observation, action_space = env.reset()
        assert isinstance(action_space, DiscreteActionSpace)
        self.assertEqual(4, action_space.n)
        # observation should contain agent(x, y) and goal(x, y)
        self.assertEqual(observation.shape, (4,))
        x, y, goal_x, goal_y = observation.tolist()
        # agent starts at arena center
        self.assertAlmostEqual(x, width / 2)
        self.assertAlmostEqual(y, height / 2)
        # goal should be within bounds
        self.assertTrue(0 <= goal_x <= width)
        self.assertTrue(0 <= goal_y <= height)

        # Test position change
        result = env.step(torch.tensor(0))
        x, y, _, _ = result.observation.tolist()
        self.assertAlmostEqual(x, observation[0].item() + 1, places=6)
        self.assertAlmostEqual(y, observation[1].item(), places=6)
        self.assertFalse(result.truncated)
        observation = result.observation

        result = env.step(torch.tensor(1))
        x, y, _, _ = result.observation.tolist()
        self.assertAlmostEqual(x, observation[0].item(), places=6)
        self.assertAlmostEqual(y, observation[1].item() + 1, places=6)
        self.assertFalse(result.truncated)
        observation = result.observation

        result = env.step(torch.tensor(2))
        x, y, _, _ = result.observation.tolist()
        self.assertAlmostEqual(x, observation[0].item() - 1, places=6)
        self.assertAlmostEqual(y, observation[1].item(), places=6)
        self.assertFalse(result.truncated)
        observation = result.observation

        result = env.step(torch.tensor(3))
        x, y, _, _ = result.observation.tolist()
        self.assertAlmostEqual(x, observation[0].item(), places=6)
        self.assertAlmostEqual(y, observation[1].item() - 1, places=6)
        self.assertFalse(result.truncated)

        # Test win reward and terminate
        env._agent_position = (goal_x, goal_y - 1)
        result = env.step(torch.tensor(1))
        self.assertEqual(result.reward, 0)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        # Test not win reward and not terminate
        env._agent_position = (goal_x - 10, goal_y - 10)
        result = env.step(torch.tensor(1))
        self.assertEqual(result.reward, -1)
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        # Test not win reward and terminate
        env._agent_position = (goal_x - 10, goal_y - 10)
        env._step_count = env._max_episode_duration
        result = env.step(torch.tensor(1))
        self.assertEqual(result.reward, -1)
        self.assertTrue(result.truncated)


class TestContinuousSparseRewardEnvironment(unittest.TestCase):
    def test_continuous_actions(self) -> None:
        width, height = 50, 50
        max_ep = 5
        env = ContinuousSparseRewardEnvironment(
            width=width, height=height, reward_distance=0.5, max_episode_duration=max_ep
        )

        # Initial reset
        observation, action_space = env.reset(seed=0)
        self.assertIsInstance(action_space, BoxActionSpace)
        self.assertEqual(observation.shape, (4,))
        x, y, goal_x, goal_y = observation.tolist()
        self.assertAlmostEqual(x, width / 2)
        self.assertAlmostEqual(y, height / 2)
        self.assertTrue(0 <= goal_x <= width)
        self.assertTrue(0 <= goal_y <= height)

        # Move by a small delta
        result = env.step(torch.tensor([0.2, -0.2]))
        x2, y2, _, _ = result.observation.tolist()
        self.assertAlmostEqual(x2, x + 0.2, places=5)
        self.assertAlmostEqual(y2, y - 0.2, places=5)
        self.assertEqual(result.reward, -1.0)
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)

        # Force a win
        env._agent_position = (goal_x - 0.1, goal_y)
        result = env.step(torch.tensor([0.1, 0.0]))
        self.assertEqual(result.reward, 0.0)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)

        # Force truncation
        observation, _ = env.reset(seed=1)
        env._agent_position = (0.0, 0.0)
        env._step_count = max_ep
        result = env.step(torch.tensor([0.0, 0.0]))
        self.assertEqual(result.reward, -1.0)
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)


class TestDiscreteSparseRewardEnvironmentInvalidActions(unittest.TestCase):
    def test_invalid_actions(self) -> None:
        env = DiscreteSparseRewardEnvironment(
            width=10, height=10, reward_distance=1, step_size=1, action_count=4
        )
        env.reset()

        with self.assertRaises(ValueError):
            env.step(torch.tensor([5]))  # index out of range

        with self.assertRaises(ValueError):
            env.step(torch.tensor([-1]))  # negative index

        with self.assertRaises(ValueError):
            env.step(torch.tensor([[0]]))  # invalid shape