# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import copy
import unittest
from typing import Iterator

import torch
import torch.testing as tt
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
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
            action_space_to_score=DiscreteActionSpace(actions=list(batch.action)),
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
        # the 1st arm occurred 10 times more than 2nd arm
        self.assertAlmostEqual(sigma_ratio, 10.0**0.5, delta=0.01)

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

    def test_unobserved_actions(self) -> None:
        """
        Simulates an application with 78 features and 24 actions,
        where only the first 5 actions are observed,
        with a randomly generated ground truth model,
        and checks whether a learned LinearBandit agrees with the ground truth model.

        Note: while this test verifies several interesting aspects of LinearBandit,
        it also reveals how poor of a model it is.
        The action selection for a given state s is actually independent in the
        state features, since they add the same additive
        term to the score of all actions:
        argmax_action dot_product(coefs, ((state, action))) =
        argmax_action dot_product(coefs[:state_dim], state)  <-- constant in action
                    + dot_product(coefs[state_dim: ], action) =
        argmax_action dot_product(coefs[state_dim: ], action)
        """
        state_dim = 78
        number_of_actions = 24
        number_of_observed_actions = 5
        action_space: DiscreteActionSpace = DiscreteActionSpace(
            [torch.tensor([i]) for i in range(number_of_actions)]
        )
        number_of_observed_features = state_dim + number_of_observed_actions

        feature_dim = state_dim + number_of_actions
        zero_feature_indices = list(range(number_of_observed_features, feature_dim))

        true_coefs = torch.zeros(feature_dim + 1)  # +1 for intercept

        # set non-zero coefficients
        intercept = feature_dim * 0.5
        true_coefs[0] = intercept  # intercept
        for i in range(1, number_of_observed_features + 1):
            if (
                i - 1 not in zero_feature_indices
            ):  # -1 because true_coefs[0] is the intercept
                true_coefs[i] = torch.rand(1).item() * 2 - 1

        ground_truth = LinearBandit(
            feature_dim=feature_dim,
            exploration_module=UCBExploration(alpha=0),
            l2_reg_lambda=1e-8,
            action_representation_module=OneHotActionTensorRepresentationModule(
                number_of_actions
            ),
            initial_coefs=true_coefs,
        )

        # Create a class to hold the action counts to avoid nonlocal issues
        class ActionCounter:
            def __init__(self, num_actions: int):
                self.counts = torch.zeros(num_actions)

            def update(self, actions: torch.Tensor) -> None:
                for a in actions:
                    self.counts[a.item()] += 1

        # Initialize counters for action occurrences during training
        training_counter = ActionCounter(number_of_actions)

        # Generator of TransitionBatch from ground truth
        def generate_batch(
            number_of_samples: int,
            state_dim: int,
            ground_truth: LinearBandit,
            counter: ActionCounter,
        ) -> Iterator[TransitionBatch]:
            # generate random state
            state = torch.rand(number_of_samples, state_dim)

            scores = ground_truth.get_scores(
                subjective_state=state,
                action_space_to_score=action_space,
                exploit=True,
            )
            action = torch.argmax(scores, dim=1).unsqueeze(-1)

            # Update action counts
            counter.update(action)

            reward = scores.gather(1, action)
            action_representation = ground_truth.action_representation_module(action)

            # generate weight
            weight = torch.ones(number_of_samples, 1)

            yield TransitionBatch(
                state=state,
                action=action_representation,
                reward=reward,
                weight=weight,
            )

        # Train a new LinearBandit with generated batches
        new_policy_learner = LinearBandit(
            feature_dim=feature_dim,
            exploration_module=UCBExploration(alpha=0.9),
            l2_reg_lambda=1e-8,
            action_representation_module=OneHotActionTensorRepresentationModule(
                number_of_actions
            ),
        )

        num_batches = 3000
        batch_size = 100

        coefficients_print_interval = int(num_batches / 10) + 1
        for i, _ in enumerate(range(num_batches)):
            batch = next(
                generate_batch(batch_size, state_dim, ground_truth, training_counter)
            )
            new_policy_learner.learn_batch(batch)
            if i % coefficients_print_interval == 0:
                # print coefficients
                print(f"True  coefs: {true_coefs}")
                print(f"Model coefs: {new_policy_learner.model.coefs}")

        # Check if the learned coefficients are close to the true coefficients
        learned_coefs = new_policy_learner.model.coefs
        # print message if not close
        if not torch.allclose(learned_coefs, true_coefs, atol=0.1, rtol=0.0):
            print("\n**** Coefficients are not close to true coefficients ****")
            print(f"True  coefs: {true_coefs}")
            print(f"Model coefs: {learned_coefs}")
            print(f"Diff  coefs: {true_coefs - learned_coefs}")
        else:
            print("Model coefs are close to true coefs")

        # Print training action distribution
        print("\nTraining action distribution:")
        for action_idx in range(number_of_actions):
            count = training_counter.counts[action_idx].item()
            percentage = (count / (num_batches * batch_size)) * 100
            print(f"Action {action_idx}: {count:.0f} occurrences ({percentage:.2f}%)")

        # Now check if ground truth and learned model agree on exploit actions
        # for a set of random states
        num_test_states = 100
        test_states = torch.rand(num_test_states, state_dim)

        # Get exploit actions from ground truth model
        ground_truth_scores = ground_truth.get_scores(
            subjective_state=test_states,
            action_space_to_score=action_space,
            exploit=True,
        )
        ground_truth_actions = torch.argmax(ground_truth_scores, dim=1)

        # Get exploit actions from learned model
        learned_scores = new_policy_learner.get_scores(
            subjective_state=test_states,
            action_space_to_score=action_space,
            exploit=True,
        )
        learned_actions = torch.argmax(learned_scores, dim=1)

        # Count occurrences of each action in evaluation
        gt_eval_action_counts = torch.zeros(number_of_actions)
        learned_eval_action_counts = torch.zeros(number_of_actions)

        for a in ground_truth_actions:
            gt_eval_action_counts[a.item()] += 1

        for a in learned_actions:
            learned_eval_action_counts[a.item()] += 1

        # Print evaluation action distributions
        print("\nEvaluation action distribution:")
        print("Action | Ground Truth | Learned Model")
        print("-----------------------------------")
        for action_idx in range(number_of_actions):
            gt_count = gt_eval_action_counts[action_idx].item()
            learned_count = learned_eval_action_counts[action_idx].item()
            gt_percentage = (gt_count / num_test_states) * 100
            learned_percentage = (learned_count / num_test_states) * 100
            print(
                f"  {action_idx:<2}  |    {gt_count:>3.0f} ({gt_percentage:>5.1f}%) | "
                f"   {learned_count:>3.0f} ({learned_percentage:>5.1f}%)"
            )

        # Calculate agreement percentage
        agreement = (ground_truth_actions == learned_actions).float().mean().item()
        agreement_percentage = agreement * 100

        # We expect a high agreement rate (at least 99%)
        print(f"\nAction agreement percentage: {agreement_percentage:.2f}%")
        error_msg = (
            f"Ground truth and learned model only agree on "
            f"{agreement_percentage:.2f}% of actions"
        )
        self.assertGreaterEqual(agreement_percentage, 99.0, error_msg)
