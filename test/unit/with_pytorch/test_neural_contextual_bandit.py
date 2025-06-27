# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.policy_learners.contextual_bandits.neural_bandit import NeuralBandit
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestNeuralContextualBandit(unittest.TestCase):
    def test_basic(self) -> None:
        feature_dim = 15
        action_dim = feature_dim - 3
        batch_size = feature_dim * 4  # it is important to have enough data for training
        policy_learner = NeuralBandit(
            feature_dim=feature_dim,
            hidden_dims=[16, 16],
            learning_rate=0.01,
            exploration_module=NoExploration(),
        )
        state = torch.randn(batch_size, 3)
        action = torch.randn(batch_size, action_dim)
        batch = TransitionBatch(
            state=state,
            action=action,
            # y = sum of state + sum of action
            reward=state.sum(-1) + action.sum(-1),
            weight=torch.randn(batch_size),
        )
        action_space = DiscreteActionSpace(actions=list(batch.action))

        # TEST LEARN
        losses = []
        for _ in range(1000):
            losses.append(policy_learner.learn_batch(batch)["loss"])

        self.assertGreater(1e-2, losses[-1])

        scores = policy_learner.get_scores(
            subjective_state=batch.state,
            action_space_to_score=action_space,
        )
        # shape should be batch_size, action_count
        self.assertEqual(scores.shape, (batch.state.shape[0], batch.action.shape[0]))

        # TEST ACT
        actions = policy_learner.act(
            subjective_state=batch.state, available_action_space=action_space
        )
        self.assertEqual(actions.shape, (len(batch), action_dim))
