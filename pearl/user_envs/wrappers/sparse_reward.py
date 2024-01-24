# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module not found.")
import math


class PendulumSparseRewardWrapper(gym.Wrapper):
    r"""Sparse Reward wrapper for the Pendulum environment."""

    def __init__(self, env):
        super(PendulumSparseRewardWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        sparse_reward = self.sparse_reward()
        return obs, sparse_reward, done, truncated, info

    def sparse_reward(self):
        th, thdot = self.env.state
        threshold = 15 / 180 * np.pi
        if (
            self.angle_normalize(th) < threshold
            and self.angle_normalize(th) > -threshold
        ):
            sparse_reward = 1
        else:
            sparse_reward = 0
        return sparse_reward

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi


class AcrobotSparseRewardWrapper(gym.Wrapper):
    r"""Sparse Reward wrapper for the Acrobot environment."""

    def __init__(self, env):
        super(AcrobotSparseRewardWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        sparse_reward = self.sparse_reward()
        return obs, sparse_reward, done, truncated, info

    def sparse_reward(self):
        s = self.env.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.0)


class MountainCarSparseRewardWrapper(gym.Wrapper):
    r"""Sparse Reward wrapper for the Mountain Car environment."""

    def __init__(self, env):
        super(MountainCarSparseRewardWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        sparse_reward = self.sparse_reward()
        return obs, sparse_reward, done, truncated, info

    def sparse_reward(self):
        position, velocity = self.env.state
        return bool(
            position >= self.env.goal_position and velocity >= self.env.goal_velocity
        )


class PuckWorldSparseRewardWrapper(gym.Wrapper):
    r"""Sparse Reward wrapper for the Mountain Car environment."""

    def __init__(self, env):
        super(PuckWorldSparseRewardWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        sparse_reward = self.sparse_reward()
        return obs, sparse_reward, done, truncated, info

    def sparse_reward(self):
        x = self.env.get_ob(self.env.game.getGameState())
        px = x[0]
        py = x[1]
        gx = x[4]
        gy = x[5]
        bx = x[6]
        by = x[7]
        dx = px - gx
        dy = py - gy
        dist_to_good = math.sqrt(dx * dx + dy * dy)

        dx = px - bx
        dy = py - by
        dist_to_bad = math.sqrt(dx * dx + dy * dy)
        reward = 0.0
        gr = self.env.game.CREEP_GOOD["radius"]
        br = self.env.game.CREEP_BAD["radius_outer"]
        if dist_to_good < gr:
            reward += 1.0 * (gr - dist_to_good) / float(gr)

        if dist_to_bad < br:
            reward += 2.0 * (dist_to_bad - br) / float(br)
        return reward / 1000
