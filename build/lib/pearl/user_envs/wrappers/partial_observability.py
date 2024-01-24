# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors

from abc import abstractmethod

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module not found.")


class PartialObservableWrapper(gym.Wrapper):
    def __init__(self, env, time_between_two_valid_obs=1):
        super(PartialObservableWrapper, self).__init__(env)
        self.env.number_of_steps = 0
        self.time_between_two_valid_obs = time_between_two_valid_obs

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self.env.number_of_steps += 1
        return self.observation(observation), reward, done, truncated, info

    @abstractmethod
    def observation(self, observation):
        raise NotImplementedError


class CartPolePartialObservableWrapper(PartialObservableWrapper):
    r"""Observation wrapper that make CartPole environment partial observable."""

    def __init__(self, env, time_between_two_valid_obs=1):
        super(CartPolePartialObservableWrapper, self).__init__(
            env, time_between_two_valid_obs
        )
        high = np.array(
            [self.x_threshold * 2, self.theta_threshold_radians * 2, 1.0],
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def observation(self, observation):
        if self.env.number_of_steps % self.time_between_two_valid_obs != 0:
            return np.zeros(3, dtype=np.float32)
        else:
            return np.array([observation[0], observation[2], 1.0], dtype=np.float32)


class AcrobotPartialObservableWrapper(PartialObservableWrapper):
    r"""Observation wrapper that make Acrobat environment partial observable."""

    def __init__(self, env, time_between_two_valid_obs=1):
        super(AcrobotPartialObservableWrapper, self).__init__(
            env, time_between_two_valid_obs
        )
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def observation(self, observation):
        if self.env.number_of_steps % self.time_between_two_valid_obs != 0:
            return np.zeros(5, dtype=np.float32)
        else:
            return np.array(
                [observation[0], observation[1], observation[2], observation[3], 1.0],
                dtype=np.float32,
            )


class PendulumPartialObservableWrapper(PartialObservableWrapper):
    r"""Observation wrapper that make Pendulum environment partial observable."""

    def __init__(self, env, time_between_two_valid_obs=1):
        super(PendulumPartialObservableWrapper, self).__init__(
            env, time_between_two_valid_obs
        )
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def observation(self, observation):
        if self.env.number_of_steps % self.time_between_two_valid_obs != 0:
            return np.zeros(3, dtype=np.float32)
        else:
            return np.array([observation[0], observation[1], 1.0], dtype=np.float32)


class MountainCarPartialObservableWrapper(PartialObservableWrapper):
    r"""Observation wrapper that make MountainCar environment partial observable."""

    def __init__(self, env, time_between_two_valid_obs=1):
        super(MountainCarPartialObservableWrapper, self).__init__(
            env, time_between_two_valid_obs
        )
        high = np.array([self.env.max_position, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def observation(self, observation):
        if self.env.number_of_steps % self.time_between_two_valid_obs != 0:
            return np.zeros(2, dtype=np.float32)
        else:
            return np.array([observation[0], 1.0], dtype=np.float32)


class PuckWorldPartialObservableWrapper(PartialObservableWrapper):
    r"""Observation wrapper that make PuckWorld environment partial observable."""

    def __init__(self, env):
        super(PuckWorldPartialObservableWrapper, self).__init__(env)
        high = np.array([np.inf, np.inf, 3], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def observation(self, observation):
        idx = self.env.number_of_steps % 4
        return np.array(
            [
                observation[2 * idx],
                observation[2 * idx + 1],
                idx,
            ],
            dtype=np.float32,
        )
