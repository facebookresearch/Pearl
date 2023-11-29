#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.utils.instantiations.environments.contextual_bandit_linear_synthetic_environment import (
    ContextualBanditLinearSyntheticEnvironment,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestContextualBanditEnvironment(unittest.TestCase):
    def setUp(self, number_of_actions: int = 2, observation_dim: int = 3) -> None:
        self.env = ContextualBanditLinearSyntheticEnvironment(
            action_space=DiscreteActionSpace(
                actions=list(torch.arange(number_of_actions).view(-1, 1))
            ),
            observation_dim=observation_dim,
        )
        self.number_of_actions = number_of_actions
        self.observation_dim = observation_dim

    def test_contextual_bandit_environment_reset(self) -> None:
        assert isinstance(self.env.action_space, DiscreteActionSpace)
        action_space: DiscreteActionSpace = self.env.action_space
        assert self.env.features_of_all_arms.shape == (
            action_space.n,
            self.env.arm_feature_vector_dim,
        )
        observation, action_space = self.env.reset()
        assert observation.shape == torch.Size([self.env.observation_dim])
        assert action_space.n == self.number_of_actions
        if action_space.n > 0:
            reward = self.env.get_reward(action=torch.tensor(0))
            # pyre-fixme[16]: `Number` has no attribute `shape`.
            assert reward.shape == torch.Size([1])  # reward is scalar
