#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.replay_buffers.sequential_decision_making.fifo_on_policy_replay_buffer import (
    FIFOOnPolicyReplayBuffer,
)

from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class TestFifoBuffer(unittest.TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    def setUp(self):
        self.batch_size = 3
        state_dim = 10
        action_dim = 3
        self.states = torch.rand(self.batch_size, state_dim)
        self.actions = torch.randint(action_dim, (self.batch_size,))
        self.rewards = torch.rand(
            self.batch_size,
        )
        self.next_states = torch.rand(self.batch_size, state_dim)
        # pyre-fixme[6]: For 1st argument expected `List[typing.Any]` but got `range`.
        self.action_space = DiscreteActionSpace(range(action_dim))
        self.curr_available_actions = self.action_space
        self.next_available_actions = self.action_space
        self.done = torch.randint(2, (self.batch_size,))

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
            self.action_space,
            False,
        )
        # push S1 A1 R1 S2
        replay_buffer.push(
            self.next_states[0],
            self.actions[1],
            self.rewards[1],
            self.next_states[1],
            self.curr_available_actions,
            self.next_available_actions,
            self.action_space,
            False,
        )
        # expect S0 A0 R0 S1 A1 returned from sample
        batch = replay_buffer.sample(1)
        self.assertTrue(
            torch.equal(
                batch.state,
                self.states[0].view(1, -1),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.action,
                torch.tensor([self.actions[0]]),
            )
        )
        self.assertTrue(torch.equal(batch.reward, torch.tensor([self.rewards[0]])))
        self.assertTrue(
            torch.equal(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Optional[Tensor]`.
                batch.next_state,
                self.next_states[0].view(1, -1),
            )
        )
        self.assertTrue(
            torch.equal(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Optional[Tensor]`.
                batch.next_action,
                torch.tensor([self.actions[1]]),
            )
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
            self.action_space,
            True,
        )
        # expect one sample returned
        batch = replay_buffer.sample(1)
        self.assertTrue(batch.done[0])
