#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch

from pearl.core.common.replay_buffer.hindsight_experience_replay_buffer import (
    HindsightExperienceReplayBuffer,
)
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class TestHindsightExperienceReplayBuffer(unittest.TestCase):
    def test_basic(self):
        """
        Test setup:
        - assume 2 x 2 box, step size is 1, 4 actions: up, down, left, right
        - reach goal get reward 0, before that all -1
        """
        # simulating a failed episide:
        # (0, 0) -> (0, 1) -> (0, 0) -> (1, 0)
        states = [
            torch.Tensor([0, 0]),
            torch.Tensor([0, 1]),
            torch.Tensor([0, 0]),
            torch.Tensor([1, 0]),
        ]
        goal = torch.Tensor([1, 1])
        action_to_next_states = {
            action: states[action + 1] for action in range(len(states) - 1)
        }

        def reward_fn(state, action):
            goal = state[-2:]
            next_state = action_to_next_states[action]
            if torch.all(torch.eq(next_state, goal)):
                return 0
            return -1

        rb = HindsightExperienceReplayBuffer(
            capacity=10, goal_dim=2, reward_fn=reward_fn
        )

        action_space = DiscreteActionSpace(range(4))
        for i in range(len(states) - 1):
            rb.push(
                state=torch.cat([states[i], goal], dim=0),
                action=i,
                reward=-1,
                next_state=torch.cat([states[i + 1], goal], dim=0),
                curr_available_actions=action_space,
                next_available_actions=action_space,
                action_space=action_space,
                done=i == len(states) - 2,
            )

        self.assertEqual(len(rb), 2 * len(states) - 2)
        batch = rb.sample(2 * len(states) - 2)

        # check if batch has a reward as 0
        self.assertTrue(torch.any(batch.reward == 0))
        # check if batch has append additional goal, which is final state
        additional_goal_count = 0
        original_goal_count = 0
        for state in batch.state:
            # states[-1] is our additional goal
            if torch.all(torch.eq(state[-2:], states[-1])):
                additional_goal_count += 1
            if torch.all(torch.eq(state[-2:], goal)):
                original_goal_count += 1
        self.assertEqual(3, additional_goal_count)
        self.assertEqual(3, original_goal_count)

        # check cache is cleared after trajectory is done
        self.assertEqual(0, len(rb._trajectory))

        # check for same transition, goal in state and next state should stay the same
        for i in range(2 * len(states) - 2):
            self.assertTrue(
                torch.all(torch.eq(batch.state[i][-2:], batch.next_state[i][-2:]))
            )
