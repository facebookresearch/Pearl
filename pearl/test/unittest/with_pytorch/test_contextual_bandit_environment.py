#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import gym
import torch
from pearl.contextual_bandits.contextual_bandit_linear_environment import (
    ContextualBanditLinearEnvironment,
)


class TestContextualBanditEnvironment(unittest.TestCase):
    def setUp(self, number_of_actions: int = 2) -> None:
        self.env = ContextualBanditLinearEnvironment(
            action_space=gym.spaces.Discrete(number_of_actions)
        )
        self.number_of_actions = number_of_actions

    def test_contextual_bandit_environment_reset(self) -> None:
        obs, action_space = self.env.reset()
        assert obs.shape[0] == action_space.n
        assert action_space.n == self.number_of_actions
        assert self.env.features_of_all_arms.shape == (
            action_space.n,
            self.env.feature_dim,
        )

    def test_contextual_bandit_environment_get_reward(self, action: int = 0):
        reward = self.env.get_reward(action=action)
        assert reward.shape == torch.Size([1])  # reward is scalar
