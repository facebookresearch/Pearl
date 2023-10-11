#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.neural_networks.common.utils import init_weights
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.utils.functional_utils.learning.nn_learning_utils import (
    optimize_twin_critics_towards_target,
)


class TestTwinCritic(unittest.TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    def setUp(self):
        self.state_dim = 20
        self.action_dim = 10
        self.batch_size = 128

    # pyre-fixme[3]: Return type must be annotated.
    def test_twin_critic(self):
        twin_critics = TwinCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=[10, 10],
            init_fn=init_weights,
        )
        state_batch = torch.randn(self.batch_size, self.state_dim)
        action_batch = torch.randn(self.batch_size, self.action_dim)
        optimizer = torch.optim.AdamW(twin_critics.parameters(), lr=1e-3)
        optimize_twin_critics_towards_target(
            twin_critics,
            optimizer,
            state_batch,
            action_batch,
            torch.randn(self.batch_size),
        )
        twin_critics.get_twin_critic_values(state_batch, action_batch)
