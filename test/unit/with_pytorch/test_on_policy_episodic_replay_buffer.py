#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.replay_buffers.sequential_decision_making.on_policy_episodic_replay_buffer import (
    OnPolicyEpisodicReplayBuffer,
)
from pearl.replay_buffers.transition import TransitionBatch

from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class TestOnPolicyEpisodicReplayBuffer(unittest.TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    def setUp(self):
        self.batch_size = 3
        self.capacity = 10

        self.rewards = [4.0, 6.0, 5.0]
        self.returns = [15.0, 11.0, 5.0]  # list of returns
        self.trajectory_len = len(self.rewards)
        self.action_size = 3
        # pyre-fixme[6]: For 1st argument expected `List[typing.Any]` but got `range`.
        self.action_space = DiscreteActionSpace(range(self.action_size))

    def test_push_incomplete_trajectory(self) -> None:
        # test that the buffer is empty and can't be sampled from until 1st complete trajectory is added
        replay_buffer = OnPolicyEpisodicReplayBuffer(self.capacity)
        for i in range(3):
            replay_buffer.push(
                state=torch.tensor([i]),
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
                state=torch.tensor([i]),
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
        # pyre-fixme
        returns_buffer = batch.cum_reward[order]
        self.assertTrue(torch.equal(returns_buffer, torch.tensor(self.returns)))

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
                state=torch.tensor([i]),
                action=torch.tensor(i),
                reward=self.rewards[i],
                next_state=None,
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                action_space=self.action_space,
                done=(i == (self.trajectory_len - 1)),
            )

        rewards_2 = [11.0, 8.0]
        returns_2 = [19.0, 8.0]
        states_2 = [4.0, 5.0]
        trajectory_len_2 = len(rewards_2)

        # push 2nd trajectory
        for i in range(trajectory_len_2):
            replay_buffer.push(
                state=torch.tensor([states_2[i]]),
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
        # pyre-fixme
        returns_buffer = batch.cum_reward[order]
        self.assertTrue(
            torch.equal(
                returns_buffer[0 : self.trajectory_len],
                torch.tensor(self.returns),
            )
        )  # 1st trajectory
        self.assertTrue(
            torch.equal(returns_buffer[self.trajectory_len :], torch.tensor(returns_2))
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

    def test_discounted_factor(self) -> None:
        replay_buffer = OnPolicyEpisodicReplayBuffer(self.capacity, 0.5)
        for i in range(self.trajectory_len):
            # cumulated return:
            # 4 + 8.5 * 0.5 --> state 0
            # 0.5*5+6=8.5 --> state 1
            # 5 --> state 2
            replay_buffer.push(
                state=torch.tensor([i]),
                action=torch.tensor(i),
                reward=self.rewards[i],
                next_state=None,
                curr_available_actions=self.action_space,
                next_available_actions=self.action_space,
                action_space=self.action_space,
                done=(i == (self.trajectory_len - 1)),
            )

        batch = replay_buffer.sample(self.trajectory_len)
        for i in range(len(batch.action)):
            if batch.state[i] == 0:
                # pyre-fixme
                self.assertEqual(4 + 8.5 * 0.5, batch.cum_reward[i])
            if batch.state[i] == 1:
                self.assertEqual(8.5, batch.cum_reward[i])
            if batch.state[i] == 2:
                self.assertEqual(5, batch.cum_reward[i])
