# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
import numpy as np
import torch

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.box import BoxSpace


class DummyBoxGymEnv(gym.Env):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )
        self._shape = shape

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self._shape, dtype=np.uint8), {}

    def step(self, action):
        return np.zeros(self._shape, dtype=np.uint8), 0.0, True, False, {}


class TestBoxSpaceND(unittest.TestCase):
    def _run_shape_test(self, shape: tuple[int, ...]):
        env = GymEnvironment(DummyBoxGymEnv(shape))
        self.assertIsInstance(env.observation_space, BoxSpace)
        self.assertEqual(env.observation_space.shape, torch.Size(shape))
        obs, _ = env.reset()
        self.assertEqual(obs.shape, shape)
        step_result = env.step(torch.tensor(0))
        self.assertEqual(step_result.observation.shape, shape)

    def test_box_space_3d_observation(self) -> None:
        self._run_shape_test((2, 3, 4))

    def test_box_space_4d_observation(self) -> None:
        self._run_shape_test((2, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()