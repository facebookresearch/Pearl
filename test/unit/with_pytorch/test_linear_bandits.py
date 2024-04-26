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
import torch.testing as tt
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.policy_learners.contextual_bandits.linear_bandit import LinearBandit
from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (  # noqa: E501
    ThompsonSamplingExplorationLinear,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    UCBExploration,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestLinearBandits(unittest.TestCase):
    def setUp(self) -> None:
        self.policy_learner = LinearBandit(
            feature_dim=4,
            exploration_module=UCBExploration(alpha=0),
            l2_reg_lambda=1e-8,
        )
        # y = sum of state + sum of action
        self.batch = TransitionBatch(
            state=torch.tensor(
                [
                    [1.0, 2.0],
                    [1.0, 3.0],
                    [2.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 2.0],
                ]
            ),
            action=torch.tensor(
                [[2.0, 2.0], [1.0, 2.0], [3.0, 2.0], [1.0, 3.0], [2.0, 2.0]]
            ),
            reward=torch.tensor([7.0, 7.0, 9.0, 9.0, 9.0]).unsqueeze(-1),
            weight=torch.tensor([1, 1, 1, 1, 1]).unsqueeze(-1),
        )
        self.policy_learner.learn_batch(self.batch)

    def test_learn(self) -> None:
        batch = self.batch
        # a single input
        tt.assert_close(
            self.policy_learner.model(
                torch.cat([batch.state[0], batch.action[0]]).unsqueeze(0),
            ),
            batch.reward[0:1],
            atol=1e-3,
            rtol=0.0,
        )
        # a batch input
        tt.assert_close(
            self.policy_learner.model(torch.cat([batch.state, batch.action], dim=1)),
            batch.reward,
            atol=1e-3,
            rtol=0.0,
        )

    def test_linear_ucb_scores(self) -> None:
        # with ucb_alpha == 0, ucb scores == rewards
        # we view action space as 1, in order to get ucb scores for given feature
        self.policy_learner.exploration_module = UCBExploration(alpha=0)
        batch = self.batch

        ucb_scores = self.policy_learner.get_scores(
            subjective_state=batch.state,
            action_space=DiscreteActionSpace(actions=list(batch.action)),
        )
        self.assertEqual(
            ucb_scores.shape, (batch.state.shape[0], batch.action.shape[0])
        )

    def test_linear_ucb_act(self) -> None:
        """
        Given a list of action features, able to return action index with highest score
        """
        policy_learner = copy.deepcopy(
            self.policy_learner
        )  # deep copy as we are going to change exploration module
        batch = self.batch
        action_space = DiscreteActionSpace(actions=list(batch.action))
        # action 2 has feature vector as 3, 2, has highest sum
        self.assertEqual(policy_learner.act(batch.state[0], action_space), 2)
        # test with batch state
        actions = policy_learner.act(batch.state, action_space)
        self.assertEqual(actions.shape, (batch.reward.shape[0],))

        policy_learner.exploration_module = UCBExploration(alpha=1)

    def test_linear_ucb_sigma(self) -> None:
        """
        Since A is init with zeros, the sigma should be proportional to 1/sqrt(number of obs)
        """
        policy_learner = LinearBandit(
            feature_dim=2,
            exploration_module=UCBExploration(alpha=1),
            l2_reg_lambda=1e-2,
        )

        # 1st arm = [1, 0]
        # 2nd arm = [0, 1]
        batch = TransitionBatch(
            state=torch.tensor(
                [
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],  # repeat 1st arm ten times
                    [0.0],
                ]
            ),
            action=torch.tensor(
                [
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [1.0],
                ]
            ),
            reward=torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0]
            ).unsqueeze(-1),
            weight=torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ).unsqueeze(-1),
        )

        policy_learner.learn_batch(batch)

        # test sigma of policy_learner (LinUCB)
        features = torch.cat([batch.state, batch.action], dim=1)
        A = policy_learner.model.A
        A_inv = torch.linalg.inv(A)
        features_with_ones = LinearRegression.append_ones(features)
        sigma = torch.sqrt(
            LinearRegression.batch_quadratic_form(features_with_ones, A_inv)
        )

        # the 2nd arm's sigma is sqrt(10) times 1st arm's sigma
        sigma_ratio = (sigma[-1] / sigma[0]).detach().item()
        self.assertAlmostEqual(
            sigma_ratio, 10.0**0.5, delta=0.01
        )  # the 1st arm occured 10 times than 2nd arm

    def test_linear_thompson_sampling_act(self) -> None:
        """
        Given a list of action features, able to return action index with highest score
        """
        policy_learner = copy.deepcopy(
            self.policy_learner
        )  # deep copy as we are going to change exploration module

        policy_learner.exploration_module = ThompsonSamplingExplorationLinear()
        batch = self.batch
        action_space = DiscreteActionSpace(actions=list(batch.action))

        # test with batch state
        selected_actions = policy_learner.act(batch.state, action_space)
        # self.assertEqual(actions.shape, batch.reward.shape)
        self.assertTrue(selected_actions.shape[0] == batch.state.shape[0])

        self.assertTrue(
            all(a in range(0, action_space.n) for a in selected_actions.tolist())
        )

    def test_linear_efficient_thompson_sampling_act(self) -> None:
        """
        Given a list of action features, able to return action index with highest score
        """
        policy_learner = copy.deepcopy(
            self.policy_learner
        )  # deep copy as we are going to change exploration module

        policy_learner.exploration_module = ThompsonSamplingExplorationLinear(
            enable_efficient_sampling=True
        )
        batch = self.batch
        action_space = DiscreteActionSpace(actions=list(batch.action))

        # test with batch state
        selected_actions = policy_learner.act(batch.state, action_space)
        # self.assertEqual(actions.shape, batch.reward.shape)
        self.assertTrue(selected_actions.shape[0] == batch.state.shape[0])

        self.assertTrue(
            all(a in range(0, action_space.n) for a in selected_actions.tolist())
        )

    def test_discounting(self) -> None:
        """
        Test discounting
        """
        policy_learner = LinearBandit(
            feature_dim=4,
            exploration_module=UCBExploration(alpha=0),
            l2_reg_lambda=1e-8,
            gamma=0.95,
            apply_discounting_interval=100.0,
        )

        num_reps = 100
        for _ in range(num_reps):
            policy_learner.learn_batch(self.batch)

        self.assertLess(
            policy_learner.model.A[0, 0].item(),
            # pyre-fixme[58]: `*` is not supported for operand types `int` and
            #  `Union[bool, float, int]`.
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            num_reps * torch.sum(self.batch.weight).item(),
        )
        self.assertLess(
            policy_learner.model._b[0].item(),
            # pyre-fixme[58]: `*` is not supported for operand types `int` and
            #  `Union[bool, float, int]`.
            num_reps * torch.sum(self.batch.reward * self.batch.weight).item(),
        )
