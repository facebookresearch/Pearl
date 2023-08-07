#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.core.common.neural_networks.nplets_critic import NpletsCritic


class TestNpletsCritic(unittest.TestCase):
    def setUp(self):
        self.state_dim = 20
        self.action_dim = 10
        self.batch_size = 128

    def target_fn(self, critic):
        state_batch = torch.randn(self.batch_size, self.state_dim)
        action_batch = torch.randn(self.batch_size, self.action_dim)
        return critic.get_batch_action_value(state_batch, action_batch)

    def test_one_critic(self):
        critic = NpletsCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[10, 10],
            learning_rate=1e-3,
            num_critics=1,
        )
        critic.optimize(self.target_fn, torch.randn(self.batch_size, 1))
        state_batch = torch.randn(self.batch_size, self.state_dim)
        action_batch = torch.randn(self.batch_size, self.action_dim)
        critic.get_q_values(state_batch, action_batch)
        critic.get_q_values(state_batch, action_batch, target=False)

    def test_twin_critic(self):
        critic = NpletsCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[10, 10],
            learning_rate=1e-3,
            num_critics=2,
        )
        critic.optimize(self.target_fn, torch.randn(self.batch_size, 1))
        state_batch = torch.randn(self.batch_size, self.state_dim)
        action_batch = torch.randn(self.batch_size, self.action_dim)
        critic.get_q_values(state_batch, action_batch)
        critic.get_q_values(state_batch, action_batch, target=False)

    def test_triplets_critic(self):
        critic = NpletsCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[10, 10],
            learning_rate=1e-3,
            num_critics=3,
        )
        critic.optimize(self.target_fn, torch.randn(self.batch_size, 1))
        state_batch = torch.randn(self.batch_size, self.state_dim)
        action_batch = torch.randn(self.batch_size, self.action_dim)
        critic.get_q_values(state_batch, action_batch)
        critic.get_q_values(state_batch, action_batch, target=False)
