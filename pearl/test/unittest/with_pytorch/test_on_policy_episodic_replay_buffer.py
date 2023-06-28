#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.core.common.replay_buffer.on_policy_episodic_replay_buffer import (
    OnPolicyEpisodicReplayBuffer,
)
from pearl.core.common.replay_buffer.transition import TransitionBatch

from pearl.utils.action_spaces import DiscreteActionSpace


class TestOnPolicyEpisodicReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.capacity = 10

        self.rewards = [4.0, 6.0, 5.0]
        self.cum_returns = [15.0, 11.0, 5.0]  # cumulative returns
        self.trajectory_len = len(self.rewards)
        self.action_size = 3
        self.action_space = DiscreteActionSpace(range(self.action_size))

    def test_push_incomplete_trajectory(self) -> None:
        # test that the buffer is empty and can't be sampled from until 1st complete trajectory is added
        replay_buffer = OnPolicyEpisodicReplayBuffer(self.capacity)
        for i in range(3):
            replay_buffer.push(
                state=torch.tensor([i]).float(),
                action=torch.tensor(i),
                reward=self.rewards[i],
                next_state=None,
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                action_space=self.action_space,
                done=False,
            )
        self.assertEqual(len(replay_buffer), 0)
        with self.assertRaises(ValueError):
            _ = replay_buffer.sample(1)

    def test_push_complete_trajectory(self) -> None:
        replay_buffer = OnPolicyEpisodicReplayBuffer(self.capacity)
        for i in range(self.trajectory_len):
            replay_buffer.push(
                state=torch.tensor([i]).float(),
                action=torch.tensor(i),
                reward=self.rewards[i],
                next_state=None,
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                action_space=self.action_space,
                done=(i == (self.trajectory_len - 1)),
            )

        self.assertEqual(len(replay_buffer), self.trajectory_len)

        batch = replay_buffer.sample(self.trajectory_len)
        self.assertIsInstance(batch, TransitionBatch)

        # sort by state (equal to step counter) because the batch was randomly ordered
        order = torch.argsort(batch.state.squeeze())

        # validate cumulative return calculation
        cum_returns_buffer = batch.reward[order]
        self.assertTrue(torch.equal(cum_returns_buffer, torch.tensor(self.cum_returns)))

        # validate terminal state indicators - 1 only for the last element
        done = batch.done[order]
        self.assertTrue(
            torch.equal(done, torch.eye(self.trajectory_len)[self.trajectory_len - 1])
        )

        # validate actions
        actions = batch.action[order]
        self.assertTrue(torch.equal(actions, torch.eye(self.action_size)))

    def test_push_2_trajectories(self) -> None:
        replay_buffer = OnPolicyEpisodicReplayBuffer(self.capacity)

        # push 1st trajectory
        for i in range(self.trajectory_len):
            replay_buffer.push(
                state=torch.tensor([i]).float(),
                action=torch.tensor(i),
                reward=self.rewards[i],
                next_state=None,
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                action_space=self.action_space,
                done=(i == (self.trajectory_len - 1)),
            )

        rewards_2 = [11.0, 8.0]
        cum_returns_2 = [19.0, 8.0]
        states_2 = [4.0, 5.0]
        trajectory_len_2 = len(rewards_2)

        # push 2nd trajectory
        for i in range(trajectory_len_2):
            replay_buffer.push(
                state=torch.tensor([states_2[i]]).float(),
                action=torch.tensor(i),
                reward=rewards_2[i],
                next_state=None,
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                action_space=self.action_space,
                done=(i == (trajectory_len_2 - 1)),
            )

        self.assertEqual(len(replay_buffer), self.trajectory_len + trajectory_len_2)

        batch = replay_buffer.sample(self.trajectory_len + trajectory_len_2)
        self.assertIsInstance(batch, TransitionBatch)

        # sort by state (equal to step counter) because the batch was randomly ordered
        order = torch.argsort(batch.state.squeeze())

        # validate cumulative return calculation
        cum_returns_buffer = batch.reward[order]
        self.assertTrue(
            torch.equal(
                cum_returns_buffer[0 : self.trajectory_len],
                torch.tensor(self.cum_returns),
            )
        )  # 1st trajectory
        self.assertTrue(
            torch.equal(
                cum_returns_buffer[self.trajectory_len :], torch.tensor(cum_returns_2)
            )
        )  # 2nd trajectory

        # validate terminal state indicators - 1 only for the last element
        done = batch.done[order]
        self.assertTrue(
            torch.equal(
                done[0 : self.trajectory_len],
                torch.eye(self.trajectory_len)[self.trajectory_len - 1],
            )
        )
        self.assertTrue(
            torch.equal(
                done[self.trajectory_len :],
                torch.eye(trajectory_len_2)[trajectory_len_2 - 1],
            )
        )

        # validate actions
        actions = batch.action[order]
        self.assertTrue(
            torch.equal(actions[0 : self.trajectory_len], torch.eye(self.action_size))
        )
        self.assertTrue(
            torch.equal(
                actions[self.trajectory_len :],
                torch.eye(self.action_size)[0:trajectory_len_2],
            )
        )
