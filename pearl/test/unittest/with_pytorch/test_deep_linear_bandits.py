#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.contextual_bandits.deep_linear_bandit import DeepLinearBandit
from pearl.contextual_bandits.disjoint_linucb_exploration import (
    DisjointLinUCBExploration,
)
from pearl.replay_buffer.transition import TransitionBatch


class TestDeepLinearBandits(unittest.TestCase):
    def test_deep_linucb(self) -> None:
        feature_dim = 15  # It is important to keep this different from hidden_dims
        batch_size = feature_dim * 4  # It is important to have enough data for training
        policy_learner = DeepLinearBandit(
            feature_dim=feature_dim,
            hidden_dims=[16, 16],
            learning_rate=0.01,
            exploration_module=DisjointLinUCBExploration(alpha=0.1),
        )
        self.assertEqual(feature_dim, policy_learner.feature_dim)
        state = torch.randn(batch_size, 3)
        action = torch.randn(batch_size, feature_dim - 3)
        batch = TransitionBatch(
            state=state,
            action=action,
            # y = sum of state + sum of action
            reward=state.sum(-1) + action.sum(-1),
            weight=torch.ones(batch_size),
        )
        losses = []
        for _ in range(1000):
            losses.append(policy_learner.learn_batch(batch)["mlp_loss"])

        self.assertGreater(1e-2, losses[-1])
        # this is to ensure e2e run to get ucb score works
        policy_learner.get_scores(torch.randn(batch_size, feature_dim))
