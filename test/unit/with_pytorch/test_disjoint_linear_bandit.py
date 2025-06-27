# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import copy
import unittest

import torch
import torch.jit
import torch.testing as tt
from pearl.policy_learners.contextual_bandits.disjoint_linear_bandit import (
    DisjointLinearBandit,
)

from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (
    ThompsonSamplingExplorationLinearDisjoint,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    DisjointUCBExploration,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestDisjointLinearBandits(unittest.TestCase):
    def setUp(self) -> None:
        action_space = DiscreteActionSpace([torch.tensor([i]) for i in range(3)])
        policy_learner = DisjointLinearBandit(
            feature_dim=2,
            action_space=action_space,
            # UCB score == rewards
            exploration_module=DisjointUCBExploration(alpha=0),
            state_features_only=True,
        )
        # y0 = x1  + x2
        # y1 = 2x1 + x2
        # y2 = 2x1 + 2x2
        batch = TransitionBatch(
            state=torch.tensor(
                [
                    [1.0, 2.0],
                    [1.0, 3.0],
                    [2.0, 3.0],
                    [1.0, 5.0],
                    [0.5, 3.0],
                    [1.8, 5.1],
                ]
            ),
            action=torch.tensor(
                [[0], [0], [1], [1], [2], [2]],
            ),
            reward=torch.tensor([3.0, 4.0, 7.0, 7.0, 7.0, 13.8]).unsqueeze(-1),
            weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unsqueeze(-1),
        )
        for _ in range(1000):
            policy_learner.learn_batch(batch)

        self.policy_learner = policy_learner
        self.batch = batch
        self.action_space = action_space

    def test_learn_batch(self) -> None:
        for i, action in enumerate(self.batch.action):
            action = action.item()
            # check if linear regression works
            tt.assert_close(
                self.policy_learner._linear_regressions[action](
                    self.batch.state[i : i + 1]
                ),
                self.batch.reward[i : i + 1],
                atol=1e-1,
                rtol=0.0,
            )

    def test_ucb_act(self) -> None:
        # deep copy as we are going to change exploration module
        policy_learner = copy.deepcopy(self.policy_learner)
        action_space = self.action_space
        batch = self.batch

        # since alpha = 0, act should return action with highest reward
        # single state
        self.assertEqual(
            2,
            policy_learner.act(
                subjective_state=torch.tensor([2.0, 3.0]), action_space=action_space
            ),
        )
        # batch state
        self.assertTrue(
            torch.all(
                policy_learner.act(
                    subjective_state=batch.state, action_space=action_space
                )
                == 2
            )
        )
        # set a different alpha value to increase weight of sigma
        policy_learner.exploration_module = DisjointUCBExploration(alpha=10000)
        # observe state [1,1] for action 1 and 2 many times,
        # this will increase sigma of action0
        # on this state, and give us act(1,1) -> 0
        batch = TransitionBatch(
            state=torch.tensor(
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ]
            ),
            action=torch.tensor(
                [[1], [2]],
            ),
            reward=torch.tensor([2.0, 3.0]).unsqueeze(-1),
            weight=torch.tensor([1.0, 1.0]).unsqueeze(-1),
        )
        for _ in range(1000):
            policy_learner.learn_batch(batch)
        self.assertEqual(
            0,
            policy_learner.act(
                subjective_state=torch.tensor([1.0, 1.0]), action_space=action_space
            ),
        )

    def test_thompson_sampling_disjoint_act(self) -> None:
        # deep copy as we are going to change exploration module
        policy_learner = copy.deepcopy(self.policy_learner)
        policy_learner.exploration_module = ThompsonSamplingExplorationLinearDisjoint()
        action_space = self.action_space
        batch = self.batch

        # test act
        self.assertTrue(
            policy_learner.act(
                subjective_state=torch.tensor([2.0, 3.0]), action_space=action_space
            )
            in range(0, action_space.n)
        )

        selected_actions = policy_learner.act(
            subjective_state=batch.state, action_space=action_space
        )
        self.assertTrue(selected_actions.shape[0] == batch.state.shape[0])

        for a in selected_actions:
            self.assertIn(a, action_space.actions_batch)

    def test_ucb_action_vector(self) -> None:
        """
        This is to test discrete action space, but each action has a action vector
        """
        state_dim = 5
        action_dim = 3
        action_count = 3
        batch_size = 10
        action_space = DiscreteActionSpace(
            actions=list(torch.randn(action_count, action_dim))
        )
        policy_learner = DisjointLinearBandit(
            feature_dim=state_dim + action_dim,
            action_space=action_space,
            exploration_module=DisjointUCBExploration(alpha=0.1),
        )
        batch = TransitionBatch(
            state=torch.randn(batch_size, state_dim),
            action=torch.randint(
                low=0, high=(action_count - 1), size=(batch_size, 1)
            ),  # this is action index
            reward=torch.randn(batch_size, 1),
            weight=torch.ones(batch_size, 1),
        )
        action = policy_learner.act(
            subjective_state=batch.state[0], action_space=action_space
        )
        # TODO: currently, act returns actions with a batch dimension
        # even if state did not have a batch dimension. This is not
        # consistent with PyTorch conventions.
        self.assertEqual(action.shape, (1, action_dim))
        action = policy_learner.act(
            subjective_state=batch.state, action_space=action_space
        )
        self.assertEqual(action.shape, (batch_size, action_dim))

    def test_traced_policy_learner(self) -> None:
        """
        Test that a traced policy learner produces the same results as the normal model
        for both individual states and state batches.
        """
        # Train the policy learner first
        policy_learner = copy.deepcopy(self.policy_learner)

        # Create example inputs for tracing
        single_state = torch.tensor([2.0, 3.0])
        batch_states = torch.tensor(
            [
                [2.0, 3.0],
                [1.0, 1.0],
                [3.0, 4.0],
            ]
        )

        # Define a wrapper function for tracing that handles the act method
        class PolicyLearnerWrapper(torch.nn.Module):
            def __init__(self, policy_learner, action_space):
                super().__init__()
                self.policy_learner = policy_learner
                self.action_space = action_space

            def forward(self, state):
                return self.policy_learner.act(
                    subjective_state=state, action_space=self.action_space
                )

        # Create the wrapper
        wrapper = PolicyLearnerWrapper(policy_learner, self.action_space)

        # Trace the model with a single state
        traced_single = torch.jit.trace(wrapper, single_state, check_trace=True)

        # Trace the model with a batch of states
        traced_batch = torch.jit.trace(wrapper, batch_states, check_trace=True)

        # Test with a single state
        original_result_single = wrapper(single_state)
        traced_result_single = traced_single(single_state)
        self.assertEqual(original_result_single, traced_result_single)

        # Test with a batch of states
        original_result_batch = wrapper(batch_states)
        traced_result_batch = traced_batch(batch_states)
        tt.assert_close(original_result_batch, traced_result_batch)

        # Test with a different single state
        different_state = torch.tensor([1.5, 2.5])
        original_result_diff = wrapper(different_state)
        traced_result_diff = traced_single(different_state)
        self.assertEqual(original_result_diff, traced_result_diff)

        # Test with a different batch of states
        different_batch = torch.tensor(
            [
                [1.5, 2.5],
                [0.5, 1.5],
                [2.5, 3.5],
            ]
        )
        original_result_diff_batch = wrapper(different_batch)
        traced_result_diff_batch = traced_batch(different_batch)
        tt.assert_close(original_result_diff_batch, traced_result_diff_batch)
