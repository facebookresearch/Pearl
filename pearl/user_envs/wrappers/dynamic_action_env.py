# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors

import torch

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module not found.")

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class DynamicActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env, reduce_action_space_cadence=4):
        super().__init__(env)
        self.number_of_steps = 0
        self.reduce_action_space_cadence = reduce_action_space_cadence

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info["available_action_space"] = DiscreteActionSpace(
            [torch.tensor([i]) for i in range(self.env.action_space.n)]
        )
        self.number_of_steps = 0
        return observation, info

    def step(self, action):
        if (
            self.number_of_steps % self.reduce_action_space_cadence == 0
            and self.number_of_steps != 0
        ):
            assert action != self.env.action_space.n - 1

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.number_of_steps += 1
        shrink = (
            1 if self.number_of_steps % self.reduce_action_space_cadence == 0 else 0
        )
        info["available_action_space"] = DiscreteActionSpace(
            [torch.tensor([i]) for i in range(self.env.action_space.n - shrink)]
        )
        return observation, reward, terminated, truncated, info
