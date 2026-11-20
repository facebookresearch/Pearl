# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


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


class HindsightSnapshotRegressionTest(unittest.TestCase):
    def test_relabel_preserves_caller_states_and_original_goals(self) -> None:
        for terminated, truncated in [(True, False), (False, True)]:
            with self.subTest(terminated=terminated, truncated=truncated):
                states = [
                    torch.tensor([0.0, 0.0, 9.0, 9.0]),
                    torch.tensor([0.0, 1.0, 9.0, 9.0]),
                    torch.tensor([1.0, 1.0, 9.0, 9.0]),
                ]
                snapshots = [state.clone() for state in states]
                space = DiscreteActionSpace([torch.tensor([0]), torch.tensor([1])])

                def reward_fn(state, action):
                    reached = snapshots[int(action.item()) + 1][:2]
                    return 0.0 if torch.equal(reached, state[-2:]) else -1.0

                buffer = HindsightExperienceReplayBuffer(10, 2, reward_fn)
                for index in range(2):
                    buffer.push(
                        states[index],
                        torch.tensor([index]),
                        -1.0,
                        terminated and index == 1,
                        truncated and index == 1,
                        space,
                        states[index + 1],
                        space,
                    )
                self.assertEqual(len(buffer), 4)
                for actual, expected in zip(states, snapshots):
                    torch.testing.assert_close(actual, expected)
                batch = buffer.sample(4)
                self.assertEqual(int((batch.state[:, -2:] == 9.0).all(dim=1).sum()), 2)
                self.assertEqual(int((batch.state[:, -2:] == 1.0).all(dim=1).sum()), 2)
                self.assertTrue(bool((batch.reward == 0.0).any()))

    def test_trajectory_snapshots_inputs_before_later_mutation(self) -> None:
        space = DiscreteActionSpace([torch.tensor([0]), torch.tensor([1])])
        buffer = HindsightExperienceReplayBuffer(10, 2, lambda state, action: -1.0)
        state = torch.tensor([0.0, 0.0, 9.0, 9.0])
        next_state = torch.tensor([0.0, 1.0, 9.0, 9.0])
        buffer.push(state, torch.tensor([0]), -1.0, False, False, space, next_state, space)
        state[:2] = 100.0
        next_state[:2] = 200.0
        buffer.push(
            torch.tensor([0.0, 1.0, 9.0, 9.0]),
            torch.tensor([1]),
            -1.0,
            True,
            False,
            space,
            torch.tensor([1.0, 1.0, 9.0, 9.0]),
            space,
        )
        batch = buffer.sample(4)
        relabeled = batch.state[(batch.state[:, -2:] == 1.0).all(dim=1)]
        self.assertEqual(
            {tuple(row.tolist()) for row in relabeled[:, :2]}, {(0.0, 0.0), (0.0, 1.0)}
        )
