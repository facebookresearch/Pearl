# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest
from typing import Any

import numpy as np
import torch
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.box import BoxSpace

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym


class MixedTupleObservationEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(3, start=1),
                gym.spaces.Box(
                    low=np.array([-1.0, 0.0], dtype=np.float32),
                    high=np.array([1.0, 2.0], dtype=np.float32),
                ),
                gym.spaces.Tuple(
                    (
                        gym.spaces.Discrete(2),
                        gym.spaces.Box(
                            low=np.array([5.0], dtype=np.float32),
                            high=np.array([6.0], dtype=np.float32),
                        ),
                    )
                ),
            )
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[tuple[object, ...], dict[str, Any]]:
        super().reset(seed=seed)
        return (2, np.array([0.25, 1.5], dtype=np.float32), (1, np.array([5.5]))), {}

    def step(
        self, action: int
    ) -> tuple[tuple[object, ...], float, bool, bool, dict[str, Any]]:
        return (
            (1, np.array([-0.5, 0.5], dtype=np.float32), (0, np.array([6.0]))),
            1.0,
            True,
            False,
            {},
        )


class TupleActionEnv(gym.Env):
    def __init__(self) -> None:
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(2), gym.spaces.Discrete(3))
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(
        self, action: tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.array([0.0], dtype=np.float32), 0.0, True, False, {}


class TestGymEnvironment(unittest.TestCase):
    def test_blackjack_tuple_observations_are_flattened(self) -> None:
        env = GymEnvironment("Blackjack-v1")

        self.assertIsInstance(env.observation_space, BoxSpace)
        self.assertEqual(env.observation_space.shape, torch.Size([3]))
        torch.testing.assert_close(
            env.observation_space.low, torch.tensor([0.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            env.observation_space.high, torch.tensor([31.0, 10.0, 1.0])
        )

        observation, action_space = env.reset(seed=0)
        self.assertIs(action_space, env.action_space)
        self.assertIsInstance(observation, torch.Tensor)
        self.assertEqual(observation.shape, torch.Size([3]))

        action_result = env.step(torch.tensor(0))
        self.assertIsInstance(action_result.observation, torch.Tensor)
        self.assertEqual(action_result.observation.shape, torch.Size([3]))

    def test_mixed_tuple_observation_space_bounds_and_values_are_flattened(
        self,
    ) -> None:
        env = GymEnvironment(MixedTupleObservationEnv())

        self.assertIsInstance(env.observation_space, BoxSpace)
        torch.testing.assert_close(
            env.observation_space.low,
            torch.tensor([1.0, -1.0, 0.0, 0.0, 5.0]),
        )
        torch.testing.assert_close(
            env.observation_space.high,
            torch.tensor([3.0, 1.0, 2.0, 1.0, 6.0]),
        )

        observation, _ = env.reset(seed=0)
        self.assertIsInstance(observation, torch.Tensor)
        torch.testing.assert_close(
            observation, torch.tensor([2.0, 0.25, 1.5, 1.0, 5.5])
        )

        action_result = env.step(torch.tensor(0))
        self.assertIsInstance(action_result.observation, torch.Tensor)
        torch.testing.assert_close(
            action_result.observation, torch.tensor([1.0, -0.5, 0.5, 0.0, 6.0])
        )

    def test_tuple_action_spaces_remain_unsupported(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "The Gym space 'Tuple' is not yet supported in Pearl.",
        ):
            GymEnvironment(TupleActionEnv())


if __name__ == "__main__":
    unittest.main()
