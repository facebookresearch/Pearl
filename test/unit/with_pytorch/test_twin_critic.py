# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.neural_networks.common.utils import init_weights
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.utils.functional_utils.learning.critic_utils import (
    twin_critic_action_value_loss,
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
        loss, _, _ = twin_critic_action_value_loss(
            state_batch=state_batch,
            action_batch=action_batch,
            expected_target_batch=torch.randn(self.batch_size),
            critic=twin_critics,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        twin_critics.get_q_values(state_batch, action_batch)
