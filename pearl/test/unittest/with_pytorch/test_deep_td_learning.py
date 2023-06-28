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
from pearl.test.utils import create_random_batch


class TestDeepTDLearning(unittest.TestCase):
    def setUp(self):
        self.batch_size = 24
        self.state_dim = 10
        self.action_dim = 3
        self.batch, self.action_space = create_random_batch(
            self.action_dim, self.state_dim, self.batch_size
        )

    def test_double_dqn(self) -> None:
        """
        For easier test, just compare double has different output as vanilla version
        both DQN start with same weights
        """
        double_dqn = DeepQLearning(
            state_dim=self.state_dim,
            action_space=self.action_space,
            hidden_dims=[3],
            training_rounds=1,
            double=True,
        )
        dqn = DeepQLearning(
            state_dim=self.state_dim,
            action_space=self.action_space,
            hidden_dims=[3],
            training_rounds=1,
            double=False,
        )
        differ = False
        for _ in range(10):
            # 10 should be large enough to see difference.
            batch1 = copy.deepcopy(self.batch)
            batch2 = copy.deepcopy(self.batch)

            double_dqn._Q.xavier_init()
            double_dqn._Q_target.xavier_init()
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
