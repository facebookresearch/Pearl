# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module not found.")


class PuckWorldSafetyWrapper(gym.Wrapper):
    r"""Safety wrapper for the PuckWorld environment.
    Small positive reward with high variance when x > width/2
    """

    def __init__(self, env, sigma=0.1):
        super(PuckWorldSafetyWrapper, self).__init__(env)
        self.sigma = sigma

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        x = obs[0]
        safety_reward = 0
        info["risky_sa"] = 0
        if x > self.env.game.width / 2:
            safety_reward = self.env.np_random.normal(0.01, self.sigma)
            info["risky_sa"] = 1
        return obs, reward + safety_reward, done, truncated, info
