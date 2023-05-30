#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest
from dataclasses import fields

import torch

import torch.nn.functional as F
from pearl.replay_buffer.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.replay_buffer.fifo_on_policy_replay_buffer import FIFOOnPolicyReplayBuffer
from pearl.replay_buffer.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffer.transition import TransitionBatch

from pearl.utils.action_spaces import DiscreteActionSpace


class TestFifoBuffer(unittest.TestCase):
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
        self.action_space = DiscreteActionSpace(range(action_dim))
        self.curr_available_actions = self.action_space
        self.next_available_actions = self.action_space
        self.done = torch.randint(2, (self.batch_size,)).float()

    def test_sample(self) -> None:
        replay_buffer = FIFOOffPolicyReplayBuffer(self.batch_size * 4)
        for i in range(self.batch_size):
            replay_buffer.push(
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.curr_available_actions,
                self.next_available_actions,
                self.action_space,
                self.done[i],
            )
        batch = replay_buffer.sample(self.batch_size)

        (
            curr_available_actions_tensor_with_padding,
            curr_available_actions_mask,
        ) = TensorBasedReplayBuffer._create_action_tensor_and_mask(
            self.action_space, self.curr_available_actions
        )

        (
            next_available_actions_tensor_with_padding,
            next_available_actions_mask,
        ) = TensorBasedReplayBuffer._create_action_tensor_and_mask(
            self.action_space, self.next_available_actions
        )
        action_tensor = F.one_hot(self.actions, num_classes=self.action_space.n)
        expected_batch = TransitionBatch(
            state=self.states,
            action=action_tensor,
            reward=self.rewards,
            next_state=self.next_states,
            curr_available_actions=next_available_actions_tensor_with_padding.expand(
                self.batch_size, -1, -1
            ),
            curr_available_actions_mask=curr_available_actions_mask.expand(
                self.batch_size, -1
            ),
            next_available_actions=next_available_actions_tensor_with_padding.expand(
                self.batch_size, -1, -1
            ),
            next_available_actions_mask=next_available_actions_mask.expand(
                self.batch_size, -1
            ),
            done=self.done,
        )
        for field in fields(expected_batch):
            field_name = field.name
            x = getattr(expected_batch, field_name)
            y = getattr(batch, field_name)
            if x is None and y is None:
                continue
            # order might be different as sample is random there
            x = x.tolist()
            y = y.tolist()
            x.sort()
            y.sort()
            self.assertEqual(x, y)

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
                TensorBasedReplayBuffer._process_single_state(self.states[0]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.action,
                TensorBasedReplayBuffer._process_single_action(
                    self.actions[0], self.action_space
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.reward,
                TensorBasedReplayBuffer._process_single_reward(self.rewards[0]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.next_state,
                TensorBasedReplayBuffer._process_single_state(self.next_states[0]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.next_action,
                TensorBasedReplayBuffer._process_single_action(
                    self.actions[1], self.action_space
                ),
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
