# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
import torch.testing as tt

from pearl.replay_buffers.sequential_decision_making.fifo_on_policy_replay_buffer import (
    FIFOOnPolicyReplayBuffer,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestFifoBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 3
        state_dim = 10
        action_dim = 3
        self.states = torch.rand(self.batch_size, state_dim)
        self.actions = torch.randint(action_dim, (self.batch_size,))
        self.rewards = torch.rand(
            self.batch_size,
        )
        self.next_states = torch.rand(self.batch_size, state_dim)
        self.action_space = DiscreteActionSpace(
            actions=[torch.tensor(i) for i in range(action_dim)]
        )
        self.curr_available_actions = self.action_space
        self.next_available_actions = self.action_space
        self.terminated = torch.randint(2, (self.batch_size,))

    def test_on_poliy_buffer_sarsa_match(self) -> None:
        """
        This test is to ensure onpolicy buffer could correctly match SARSA pair
        for single push
        """
        replay_buffer = FIFOOnPolicyReplayBuffer(self.batch_size * 4)
        # push S0 A0 R0 S1
        replay_buffer.push(
            self.states[0],
            self.actions[0],
            self.rewards[0],
            self.next_states[0],
            self.curr_available_actions,
            self.next_available_actions,
            False,
            self.action_space.n,
        )
        # push S1 A1 R1 S2
        replay_buffer.push(
            self.next_states[0],
            self.actions[1],
            self.rewards[1],
            self.next_states[1],
            self.curr_available_actions,
            self.next_available_actions,
            False,
            self.action_space.n,
        )
        # expect S0 A0 R0 S1 A1 returned from sample
        batch = replay_buffer.sample(1)
        tt.assert_close(
            batch.state,
            self.states[0].view(1, -1),
            rtol=0.0,
            atol=0.0,
        )
        tt.assert_close(
            batch.action,
            torch.tensor([self.actions[0]]),
            rtol=0.0,
            atol=0.0,
        )
        tt.assert_close(
            batch.reward, torch.tensor([self.rewards[0]]), rtol=0.0, atol=0.0
        )
        assert (batch_next_state := batch.next_state) is not None
        tt.assert_close(
            batch_next_state,
            self.next_states[0].view(1, -1),
            rtol=0.0,
            atol=0.0,
        )
        assert (batch_next_action := batch.next_action) is not None
        tt.assert_close(
            batch_next_action,
            torch.tensor([self.actions[1]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_on_poliy_buffer_ternimal_push(self) -> None:
        """
        This test is to ensure onpolicy buffer could push for terminal state
        """
        replay_buffer = FIFOOnPolicyReplayBuffer(self.batch_size * 4)
        replay_buffer.push(
            self.states[0],
            self.actions[0],
            self.rewards[0],
            self.next_states[0],
            self.curr_available_actions,
            self.next_available_actions,
            True,
            self.action_space.n,
        )
        # expect one sample returned
        batch = replay_buffer.sample(1)
        self.assertTrue(batch.terminated[0])
