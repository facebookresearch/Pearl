#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import copy
import unittest

import torch
from pearl.core.common.policy_learners.exploration_module.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.core.sequential_decision_making.policy_learners.deep_q_learning import (
    DeepQLearning,
)
from pearl.core.sequential_decision_making.policy_learners.deep_sarsa import DeepSARSA
from pearl.core.sequential_decision_making.policy_learners.double_dqn import DoubleDQN
from pearl.neural_networks.common.utils import init_weights
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class TestDeepTDLearning(unittest.TestCase):
    def setUp(self):
        self.batch_size = 24
        self.state_dim = 10
        self.action_dim = 3
        self.action_space = DiscreteActionSpace(range(self.action_dim))
        buffer = FIFOOffPolicyReplayBuffer(self.batch_size)
        for _ in range(self.batch_size):
            buffer.push(
                torch.randn(self.state_dim),
                torch.randint(self.action_dim, (1,)),
                torch.randint(1, (1,)),
                torch.randn(self.state_dim),
                self.action_space,
                self.action_space,
                self.action_space,
                False,
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
        )
        dqn = DeepQLearning(
            state_dim=self.state_dim,
            action_space=self.action_space,
            hidden_dims=[3],
            training_rounds=1,
        )
        differ = False
        for _ in range(10):
            # 10 should be large enough to see difference.
            batch1 = copy.deepcopy(self.batch)
            batch2 = copy.deepcopy(self.batch)

            double_dqn._Q.apply(init_weights)
            double_dqn._Q_target.apply(init_weights)
            double_value = double_dqn._get_next_state_values(batch1, self.batch_size)

            dqn._Q.load_state_dict(double_dqn._Q.state_dict())
            dqn._Q_target.load_state_dict(double_dqn._Q_target.state_dict())
            vanilla_value = dqn._get_next_state_values(batch2, self.batch_size)
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
        )
        sa_value = sarsa._get_next_state_values(self.batch, self.batch_size)

        self.assertEqual(sa_value.shape, (self.batch_size,))
