#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.neural_networks.common.utils import init_weights
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    twin_critic_action_value_update,
)


class TestTwinCritic(unittest.TestCase):
    def setUp(self) -> None:
        self.state_dim = 20
        self.action_dim = 10
        self.batch_size = 128

    def test_twin_critic(self) -> None:
        twin_critics = TwinCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[10, 10],
            init_fn=init_weights,
        )
        state_batch = torch.randn(self.batch_size, self.state_dim)
        action_batch = torch.randn(self.batch_size, self.action_dim)
        optimizer = torch.optim.AdamW(twin_critics.parameters(), lr=1e-3)
        twin_critic_action_value_update(
            state_batch=state_batch,
            action_batch=action_batch,
            expected_target_batch=torch.randn(self.batch_size),
            optimizer=optimizer,
            critic=twin_critics,
        )
        twin_critics.get_q_values(state_batch, action_batch)
