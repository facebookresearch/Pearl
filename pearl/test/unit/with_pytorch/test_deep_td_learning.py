#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
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
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import (
    FIFOOffPolicyReplayBuffer,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestDeepTDLearning(unittest.TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    def setUp(self):
        self.batch_size = 24
        self.state_dim = 10
        self.action_count = 3
        self.action_space = DiscreteActionSpace(
            actions=list(torch.arange(self.action_count).view(-1, 1))
        )
        self.action_representation_module = OneHotActionTensorRepresentationModule(
            max_actions=3
        )
        buffer = FIFOOffPolicyReplayBuffer(self.batch_size)
        for _ in range(self.batch_size):
            buffer.push(
                torch.randn(self.state_dim),
                self.action_space.sample(),
                # pyre-fixme[6]: For 3rd argument expected `float` but got `Tensor`.
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
        double_dqn.set_action_representation_module(self.action_representation_module)
        dqn = DeepQLearning(
            state_dim=self.state_dim,
            action_space=self.action_space,
            hidden_dims=[3],
            training_rounds=1,
        )
        dqn.set_action_representation_module(self.action_representation_module)
        differ = False
        for _ in range(10):
            # 10 should be large enough to see difference.
            batch1 = double_dqn.preprocess_batch(copy.deepcopy(self.batch))
            batch2 = dqn.preprocess_batch(copy.deepcopy(self.batch))
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
        sarsa.set_action_representation_module(self.action_representation_module)
        sa_value = sarsa._get_next_state_values(
            batch=sarsa.preprocess_batch(self.batch), batch_size=self.batch_size
        )
        self.assertEqual(sa_value.shape, (self.batch_size,))
