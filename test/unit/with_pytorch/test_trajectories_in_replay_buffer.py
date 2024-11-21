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
from pearl.replay_buffers import BasicReplayBuffer

from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestTrajectoriesInReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 3
        self.capacity = 10

        self.rewards = [4.0, 6.0, 5.0]
        self.returns = [15.0, 11.0, 5.0]  # list of returns
        self.trajectory_len = len(self.rewards)
        self.action_size = 3
        self.action_space = DiscreteActionSpace(
            actions=[torch.tensor(i) for i in range(self.action_size)]
        )

    def test_push_complete_trajectory(self) -> None:
        replay_buffer = BasicReplayBuffer(self.capacity)
        for i in range(self.trajectory_len):
            replay_buffer.push(
                state=torch.tensor([i]),
                action=torch.tensor(i),
                reward=self.rewards[i],
                next_state=torch.tensor([i]),
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                terminated=True,
                truncated=(i == (self.trajectory_len - 1)),
                max_number_actions=self.action_space.n,
            )

        self.assertEqual(len(replay_buffer), self.trajectory_len)

        batch = replay_buffer.sample(self.trajectory_len)
        self.assertIsInstance(batch, TransitionBatch)

        # sort by state (equal to step counter) because the batch was randomly ordered
        order = torch.argsort(batch.state.squeeze())

        # validate terminal state indicators - 1 only for the last element
        truncated = batch.truncated[order]
        tt.assert_close(
            truncated,
            torch.eye(self.trajectory_len)[self.trajectory_len - 1].bool(),
            rtol=0.0,
            atol=0.0,
        )

        # validate actions
        actions = batch.action[order]
        tt.assert_close(actions, torch.arange(self.action_size), rtol=0.0, atol=0.0)

    def test_push_2_trajectories(self) -> None:
        replay_buffer = BasicReplayBuffer(self.capacity)

        # push 1st trajectory
        for i in range(self.trajectory_len):
            replay_buffer.push(
                state=torch.tensor([i]),
                action=torch.tensor(i),
                reward=self.rewards[i],
                next_state=torch.tensor([i]),
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                terminated=True,
                truncated=(i == (self.trajectory_len - 1)),
                max_number_actions=self.action_space.n,
            )

        rewards_2 = [11.0, 8.0]
        states_2 = [4.0, 5.0]
        trajectory_len_2 = len(rewards_2)

        # push 2nd trajectory
        for i in range(trajectory_len_2):
            replay_buffer.push(
                state=torch.tensor([states_2[i]]),
                action=torch.tensor(i),
                reward=rewards_2[i],
                next_state=torch.tensor([i]),
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                terminated=True,
                truncated=(i == (trajectory_len_2 - 1)),
                max_number_actions=self.action_space.n,
            )

        self.assertEqual(len(replay_buffer), self.trajectory_len + trajectory_len_2)

        batch = replay_buffer.sample(self.trajectory_len + trajectory_len_2)
        self.assertIsInstance(batch, TransitionBatch)

        # sort by state (equal to step counter) because the batch was randomly ordered
        order = torch.argsort(batch.state.squeeze())

        # validate terminal state indicators - 1 only for the last element
        truncated = batch.truncated[order]
        tt.assert_close(
            truncated[0 : self.trajectory_len],
            torch.eye(self.trajectory_len)[self.trajectory_len - 1].bool(),
            rtol=0.0,
            atol=0.0,
        )
        tt.assert_close(
            truncated[self.trajectory_len :],
            torch.eye(trajectory_len_2)[trajectory_len_2 - 1].bool(),
            rtol=0.0,
            atol=0.0,
        )

        # validate actions
        actions = batch.action[order]
        tt.assert_close(
            actions[0 : self.trajectory_len],
            torch.arange(self.action_size),
            rtol=0.0,
            atol=0.0,
        )
        tt.assert_close(
            actions[self.trajectory_len :],
            torch.arange(self.action_size)[0:trajectory_len_2],
            rtol=0.0,
            atol=0.0,
        )
