# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

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
        observation, _ = self.env.reset()
        assert isinstance(observation, torch.Tensor)
        assert observation.shape == torch.Size([self.env.observation_dim])
        assert action_space.n == self.number_of_actions
        if action_space.n > 0:
            reward = self.env.get_reward(action=torch.tensor(0))
            assert isinstance(reward, torch.Tensor)
            assert reward.shape == torch.Size([1])  # reward is scalar
