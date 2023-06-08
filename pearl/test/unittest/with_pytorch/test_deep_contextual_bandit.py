#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.contextual_bandits.deep_bandit import DeepBandit
from pearl.policy_learners.exploration_module.no_exploration import NoExploration
from pearl.replay_buffer.transition import TransitionBatch


class TestDeepContextualBandit(unittest.TestCase):
    def test_deep_basic(self) -> None:
        feature_dim = 15
        batch_size = feature_dim * 4  # it is important to have enough data for training
        policy_learner = DeepBandit(
            feature_dim=feature_dim,
            hidden_dims=[16, 16],
            learning_rate=0.01,
            exploration_module=NoExploration(),
        )
        state = torch.randn(batch_size, 3)
        action = torch.randn(batch_size, feature_dim - 3)
        batch = TransitionBatch(
            state=state,
            action=action,
            # y = sum of state + sum of action
            reward=state.sum(-1) + action.sum(-1),
            weight=torch.randn(batch_size),
        )
        losses = []
        for _ in range(1000):
            losses.append(policy_learner.learn_batch(batch)["loss"])

        self.assertGreater(1e-2, losses[-1])
        self.assertTrue(
            torch.allclose(
                batch.reward,
                policy_learner.get_scores(
                    torch.cat([batch.state, batch.action], dim=1)
                ),
                atol=0.1,
            )
        )