# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import copy
import unittest

import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.neural_networks.common.utils import init_weights
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.deep_sarsa import DeepSARSA
from pearl.policy_learners.sequential_decision_making.double_dqn import DoubleDQN
from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestDeepTDLearning(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 24
        self.state_dim = 10
        self.action_count = 3
        self.action_space = DiscreteActionSpace(
            actions=list(torch.arange(self.action_count).view(-1, 1))
        )
        self.action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=3
        )
        buffer = BasicReplayBuffer(self.batch_size)
        for _ in range(self.batch_size):
            buffer.push(
                state=torch.randn(self.state_dim),
                action=self.action_space.sample(),
                reward=torch.randint(1, (1,)),
                next_state=torch.randn(self.state_dim),
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                terminated=False,
                truncated=False,
                max_number_actions=self.action_space.n,
            )
        self.batch = buffer.sample(self.batch_size)
        self.batch.next_action = self.batch.action

    def test_double_dqn(self) -> None:
        """
        For easier test, just compare double has different output as vanilla version
        both DQN start with same weights
        """
        double_dqn = DoubleDQN(
            state_dim=self.state_dim,
            action_space=self.action_space,
            hidden_dims=[3],
            training_rounds=1,
            action_representation_module=self.action_representation_module,
        )
        dqn = DeepQLearning(
            state_dim=self.state_dim,
            action_space=self.action_space,
            hidden_dims=[3],
            training_rounds=1,
            action_representation_module=self.action_representation_module,
        )
        differ = False
        for _ in range(10):
            # 10 should be large enough to see difference.
            batch1 = double_dqn.preprocess_batch(copy.deepcopy(self.batch))
            batch2 = dqn.preprocess_batch(copy.deepcopy(self.batch))
            double_dqn._Q.apply(init_weights)
            double_dqn._Q_target.apply(init_weights)
            double_value = double_dqn.get_next_state_values(batch1, self.batch_size)

            dqn._Q.load_state_dict(double_dqn._Q.state_dict())
            dqn._Q_target.load_state_dict(double_dqn._Q_target.state_dict())
            vanilla_value = dqn.get_next_state_values(batch2, self.batch_size)
            self.assertEqual(double_value.shape, vanilla_value.shape)
            differ = torch.any(double_value != vanilla_value)
            if differ:
                break
        self.assertTrue(differ)

    def test_sarsa(self) -> None:
        sarsa = DeepSARSA(
            state_dim=self.state_dim,
            action_space=self.action_space,
            hidden_dims=[3],
            training_rounds=1,
            exploration_module=EGreedyExploration(0.05),
            action_representation_module=self.action_representation_module,
        )
        sa_value = sarsa.get_next_state_values(
            batch=sarsa.preprocess_batch(self.batch), batch_size=self.batch_size
        )
        self.assertEqual(sa_value.shape, (self.batch_size,))
