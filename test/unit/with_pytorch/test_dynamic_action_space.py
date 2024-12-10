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
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.replay_buffers import BasicReplayBuffer

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestDynamicActionSpaceReplayBuffer(unittest.TestCase):
    def test_basic(self) -> None:
        """
        Test setup:
        - assume max_number_actions = 5
        - push a transition with dynamic action space of [0, 2, 4] in action space
        - after push, expect available_mask = [0, 0, 0, 1, 1] and action space to be [
            [0],
            [2],
            [4],
            [0],
            [0],
        ]
        - expect available_mask = [0, 0, 0, 1, 1] and one hot representation to be [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
        """
        state_dim = 10
        current_action_space = DiscreteActionSpace(
            [torch.tensor([i]) for i in [0, 2, 4]]
        )
        next_action_space = DiscreteActionSpace([torch.tensor([i]) for i in [0, 3]])
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=5
        )
        replay_buffer = BasicReplayBuffer(capacity=10)
        replay_buffer.push(
            state=torch.zeros(state_dim),
            action=current_action_space.sample(),
            reward=0,
            next_state=torch.zeros(state_dim),
            curr_available_actions=current_action_space,
            next_available_actions=next_action_space,
            terminated=False,
            truncated=False,
            max_number_actions=action_representation_module.max_number_actions,
        )
        batch = replay_buffer.sample(1)
        current_available_actions = batch.curr_available_actions
        current_available_actions_mask = batch.curr_unavailable_actions_mask
        self.assertIsNotNone(current_available_actions)
        tt.assert_close(
            current_available_actions,
            torch.tensor([[[0.0], [2.0], [4.0], [0.0], [0.0]]]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertIsNotNone(current_available_actions_mask)
        tt.assert_close(
            current_available_actions_mask,
            torch.tensor([[False, False, False, True, True]]),
            rtol=0.0,
            atol=0.0,
        )

        next_available_actions = batch.next_available_actions
        next_unavailable_actions_mask = batch.next_unavailable_actions_mask
        self.assertIsNotNone(next_available_actions)
        tt.assert_close(
            next_available_actions,
            torch.tensor([[[0.0], [3.0], [0.0], [0.0], [0.0]]]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertIsNotNone(next_unavailable_actions_mask)
        tt.assert_close(
            next_unavailable_actions_mask,
            torch.tensor([[False, False, True, True, True]]),
        )

        policy_learner = DeepQLearning(
            state_dim=state_dim,
            hidden_dims=[3],
            training_rounds=1,
            action_representation_module=action_representation_module,
        )

        batch = policy_learner.preprocess_batch(batch)
        current_available_actions = batch.curr_available_actions
        current_unavailable_actions_mask = batch.curr_unavailable_actions_mask
        self.assertIsNotNone(current_available_actions)
        tt.assert_close(
            current_available_actions,
            torch.tensor(
                [
                    [
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                    ]
                ]
            ).float(),
            rtol=0.0,
            atol=0.0,
        )
        self.assertIsNotNone(current_unavailable_actions_mask)
        tt.assert_close(
            current_unavailable_actions_mask,
            torch.tensor([[False, False, False, True, True]]),
            rtol=0.0,
            atol=0.0,
        )

        next_available_actions = batch.next_available_actions
        next_unavailable_actions_mask = batch.next_unavailable_actions_mask
        self.assertIsNotNone(next_available_actions)
        tt.assert_close(
            next_available_actions,
            torch.tensor(
                [
                    [
                        [1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                    ]
                ]
            ).float(),
            rtol=0.0,
            atol=0.0,
        )
        self.assertIsNotNone(next_unavailable_actions_mask)
        tt.assert_close(
            next_unavailable_actions_mask,
            torch.tensor([[False, False, True, True, True]]),
            rtol=0.0,
            atol=0.0,
        )
