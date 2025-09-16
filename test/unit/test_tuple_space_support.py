# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
import torch

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.box import BoxSpace


class DummyTupleGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(3),
            )
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return (0, 1), {}

    def step(self, action):
        return (0, 1), 0.0, True, False, {}


class TestTupleSpaceSupport(unittest.TestCase):
    def test_gym_environment_tuple_observation(self):
        env = GymEnvironment(DummyTupleGymEnv())
        self.assertIsInstance(env.observation_space, BoxSpace)
        self.assertEqual(env.observation_space.shape[0], 2)
        obs, _ = env.reset()
        self.assertEqual(obs, (0, 1))
        step_result = env.step(torch.tensor(0))
        self.assertEqual(step_result.observation, (0, 1))


if __name__ == "__main__":
    unittest.main()