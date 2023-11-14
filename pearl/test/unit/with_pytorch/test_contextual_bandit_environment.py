#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

import torch
from pearl.utils.instantiations.environments.contextual_bandit_linear_synthetic_environment import (
    ContextualBanditLinearSyntheticEnvironment,
)


class TestContextualBanditEnvironment(unittest.TestCase):
    def setUp(self, number_of_actions: int = 2, observation_dim: int = 3) -> None:
        self.env = ContextualBanditLinearSyntheticEnvironment(
            # pyre-fixme[6]: For 1st argument expected `ActionSpace` but got `Discrete`.
            action_space=gym.spaces.Discrete(number_of_actions),
            observation_dim=observation_dim,
        )
        self.number_of_actions = number_of_actions
        self.observation_dim = observation_dim

    def test_contextual_bandit_environment_reset(self) -> None:
        assert self.env.features_of_all_arms.shape == (
            # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
            self.env.action_space.n,
            self.env.arm_feature_vector_dim,
        )
        observation, action_space = self.env.reset()
        assert observation.shape == torch.Size([self.env.observation_dim])
        assert action_space.n == self.number_of_actions
        if self.env.action_space.n > 0:
            reward = self.env.get_reward(action=0)
            # pyre-fixme[16]: `Number` has no attribute `shape`.
            assert reward.shape == torch.Size([1])  # reward is scalar
