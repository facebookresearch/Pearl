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

from pearl.replay_buffers.sequential_decision_making.hindsight_experience_replay_buffer import (
    HindsightExperienceReplayBuffer,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestHindsightExperienceReplayBuffer(unittest.TestCase):
    def test_basic(self) -> None:
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
        actions = [torch.tensor([i]) for i in range(len(states) - 1)]
        action_to_next_states: dict[int, torch.Tensor] = {
            int(action.item()): states[i + 1] for i, action in enumerate(actions)
        }

        def reward_fn(state: torch.Tensor, action: torch.Tensor) -> int:
            goal = state[-2:]
            next_state = action_to_next_states[int(action.item())]
            if torch.all(torch.eq(next_state, goal)):
                return 0
            return -1

        rb = HindsightExperienceReplayBuffer(
            capacity=10, goal_dim=2, reward_fn=reward_fn
        )

        action_space = DiscreteActionSpace(
            actions=[torch.tensor([i]) for i in range(4)]
        )
        for i in range(len(states) - 1):
            rb.push(
                state=torch.cat([states[i], goal], dim=0),
                action=torch.tensor([i]),
                reward=-1,
                next_state=torch.cat([states[i + 1], goal], dim=0),
                curr_available_actions=action_space,
                next_available_actions=action_space,
                terminated=i == len(states) - 2,
                truncated=False,
                max_number_actions=action_space.n,
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

        # check cache is cleared after trajectory is terminated
        self.assertEqual(0, len(rb._trajectory))

        # check for same transition, goal in state and next state should stay the same
        assert (batch_state := batch.state) is not None
        assert (batch_next_state := batch.next_state) is not None
        for i in range(2 * len(states) - 2):
            tt.assert_close(
                batch_state[i][-2:], batch_next_state[i][-2:], rtol=0.0, atol=0.0
            )
