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
from parameterized import parameterized_class
from pearl.policy_learners.contextual_bandits.disjoint_bandit import (
    DisjointBanditContainer,
)
from pearl.policy_learners.contextual_bandits.disjoint_linear_bandit import (
    DisjointLinearBandit,
)
from pearl.policy_learners.contextual_bandits.linear_bandit import LinearBandit
from pearl.policy_learners.contextual_bandits.neural_linear_bandit import (
    NeuralLinearBandit,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.thompson_sampling_exploration import (  # noqa E501
    ThompsonSamplingExplorationLinearDisjoint,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (  # noqa E501
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

        self.assertTrue(
            all(a in range(0, action_space.n) for a in selected_actions.tolist())
        )

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
        self.assertEqual(action.shape, ())
        action = policy_learner.act(
            subjective_state=batch.state, action_space=action_space
        )
        self.assertEqual(action.shape, torch.Size([batch_size]))


@parameterized_class(
    ("bandit_class", "bandit_kwargs"),
    [
        (LinearBandit, {}),
        (NeuralLinearBandit, {"hidden_dims": [20], "learning_rate": 3e-3}),
    ],
)
class TestDisjointBanditContainerBandits(unittest.TestCase):
    def setUp(self) -> None:
        self.bandit_kwargs = self.bandit_kwargs
        self.bandit_class = self.bandit_class
        num_arms = 3
        self.action_space = DiscreteActionSpace(
            [torch.tensor([i]) for i in range(num_arms)]
        )
        feature_dim = 2
        bandit_kwargs = copy.deepcopy(self.bandit_kwargs)
        bandit_kwargs["feature_dim"] = feature_dim
        self.policy_learner = DisjointBanditContainer(
            feature_dim=feature_dim,
            arm_bandits=[self.bandit_class(**bandit_kwargs) for _ in range(num_arms)],
            exploration_module=DisjointUCBExploration(alpha=0),
            state_features_only=True,
        )
        # y0 = x1  + x2
        # y1 = 2x1 + x2
        # y2 = 2x1 + 2x2
        self.batch = TransitionBatch(
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

    def test_learn_batch(self) -> None:
        # deep copy as we are going to change exploration module
        policy_learner = copy.deepcopy(self.policy_learner)
        for _ in range(1000):
            policy_learner.learn_batch(self.batch)

        for i, action in enumerate(self.batch.action):
            action = action.item()
            # check if each arm model works
            tt.assert_close(
                policy_learner._arm_bandits[action].model(self.batch.state[i : i + 1]),
                self.batch.reward[i : i + 1],
                atol=1e-1,
                rtol=0.0,
            )

    def test_ucb_act(self) -> None:
        # TODO: the condition below is never satisfied. The test should be on self.bandit_class
        if not isinstance(self.policy_learner, LinearBandit):
            # This test is reliable only for linear bandits. NN have too much
            # variance during training
            return
        # deep copy as we are going to change exploration module
        policy_learner = copy.deepcopy(self.policy_learner)
        action_space = self.action_space
        batch = self.batch

        for _ in range(1000):
            policy_learner.learn_batch(self.batch)

        # since alpha = 0, act should return action with highest reward
        # single state
        self.assertEqual(
            2,
            policy_learner.act(
                subjective_state=torch.tensor([2.0, 3.0]),
                available_action_space=action_space,
            ),
        )
        # batch state
        self.assertTrue(
            torch.all(
                policy_learner.act(
                    subjective_state=batch.state, available_action_space=action_space
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
                subjective_state=torch.tensor([1.0, 1.0]),
                available_action_space=action_space,
            ),
        )

    def test_thompson_sampling_disjoint_act(self) -> None:
        if not isinstance(self.policy_learner, LinearBandit):
            # This test only supports linear bandits
            return

        # deep copy as we are going to change exploration module
        policy_learner = copy.deepcopy(self.policy_learner)
        policy_learner.exploration_module = ThompsonSamplingExplorationLinearDisjoint()
        action_space = self.action_space
        batch = self.batch

        # test act
        self.assertTrue(
            policy_learner.act(
                subjective_state=torch.tensor([2.0, 3.0]),
                available_action_space=action_space,
            )
            in range(0, action_space.n)
        )

        selected_actions = policy_learner.act(
            subjective_state=batch.state, available_action_space=action_space
        )
        self.assertTrue(selected_actions.shape[0] == batch.state.shape[0])

        self.assertTrue(
            all(a in range(0, action_space.n) for a in selected_actions.tolist())
        )

    def test_ucb_action_vector(self) -> None:
        """
        This is to test discrete action space, but each action has a action vector
        """
        state_dim = 5
        action_dim = 1
        action_count = 3
        batch_size = 10
        action_space = DiscreteActionSpace(
            [torch.tensor([i]) for i in range(action_count)]
        )
        bandit_kwargs = copy.deepcopy(self.bandit_kwargs)
        bandit_kwargs["feature_dim"] = state_dim + action_dim
        policy_learner = DisjointBanditContainer(
            feature_dim=state_dim + action_dim,
            arm_bandits=[
                self.bandit_class(**bandit_kwargs) for _ in range(action_count)
            ],
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
            subjective_state=batch.state[0], available_action_space=action_space
        )
        self.assertEqual(action.shape, ())
        action = policy_learner.act(
            subjective_state=batch.state, available_action_space=action_space
        )
        self.assertEqual(action.shape, torch.Size([batch_size]))

    def test_get_scores(self) -> None:
        # deep copy as we are going to change exploration module
        policy_learner = copy.deepcopy(self.policy_learner)
        alpha = 3.0
        policy_learner.exploration_module = DisjointUCBExploration(alpha=alpha)
        # action_space = self.action_space
        # batch = self.batch
        batch_size = len(self.batch)

        # get scores
        scores = policy_learner.get_scores(
            subjective_state=self.batch.state, action_space_to_score=self.action_space
        )
        self.assertEqual(scores.shape, torch.Size([batch_size, self.action_space.n]))

        # test that scores have the correct values
        features = self.batch.state
        expected_scores = []
        for i in range(self.action_space.n):
            model = policy_learner.models[i]  # model for arm i
            mus = model(features)
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            sigmas = model.calculate_sigma(features)
            expected_scores.append(mus + alpha * sigmas)
        expected_scores = torch.cat(expected_scores, dim=1)
        tt.assert_close(scores, expected_scores, atol=1e-1, rtol=0.0)

    def test_learn_batch_arm_subset(self) -> None:
        # test that learn_batch still works when the batch has a subset of arms

        policy_learner = copy.deepcopy(self.policy_learner)

        # action 0 is missing from the batch
        batch = TransitionBatch(
            state=torch.tensor(
                [
                    [2.0, 3.0],
                    [1.0, 5.0],
                    [0.5, 3.0],
                    [1.8, 5.1],
                ]
            ),
            action=torch.tensor(
                [[1], [1], [2], [2]],
            ),
            reward=torch.tensor([7.0, 7.0, 7.0, 13.8]).unsqueeze(-1),
            weight=torch.tensor([1.0, 1.0, 1.0, 1.0]).unsqueeze(-1),
        )

        # learn batch, make sure this doesn't throw an error
        policy_learner.learn_batch(batch)


class TestDisjointBanditContainerLearningFromGroundTruth(unittest.TestCase):
    def test_learning_from_ground_truth_with_some_unobserved_actions(self) -> None:
        """
        Simulates an application with 78 features and 24 actions,
        with a randomly generated ground truth model,
        and checks whether a learned disjoint linear bandit model
        agrees with the ground truth model.

        Also tests that actions with indices greater than or equal to
        unobserved_actions_first_index have large thresholds and are never selected,
        and that their scores are identical across all evaluation examples.
        """
        state_dim = 78
        number_of_actions = 24
        unobserved_actions_first_index = (
            16  # Actions >= this index should never be selected
        )
        action_space: DiscreteActionSpace = DiscreteActionSpace(
            [torch.tensor([i]) for i in range(number_of_actions)]
        )

        ground_truth_linear_bandits = []
        for action_idx in range(number_of_actions):
            true_coefs = torch.zeros(state_dim + 1)  # +1 for intercept

            if action_idx >= unobserved_actions_first_index:
                # Unobserved actions lead the corresponding linear bandits
                # to have thresholds and weights with values 0
                # (their initial values, since they are never updated).
                # We set the ground truth in the same way so that
                # we don't penalize that assumption during this test.
                # (while that assumption may be incorrect in different applications,
                # measuring that inadequacy is not the purpose of this test).
                true_coefs = torch.zeros(state_dim + 1)
            else:
                true_coefs[0] = 0  # intercept
                for i in range(1, state_dim + 1):
                    true_coefs[i] = torch.rand(1).item() * 2 - 1

            ground_truth_linear_bandit = LinearBandit(
                feature_dim=state_dim,
                initial_coefs=true_coefs,
            )
            ground_truth_linear_bandits.append(ground_truth_linear_bandit)

        ground_truth = DisjointBanditContainer(
            feature_dim=state_dim,
            arm_bandits=ground_truth_linear_bandits,
            exploration_module=DisjointUCBExploration(alpha=0),
            state_features_only=True,
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
            ground_truth: DisjointBanditContainer,
            counter: ActionCounter,
            unobserved_actions_first_index: int,
        ) -> Iterator[TransitionBatch]:
            # generate random state
            state = torch.rand(number_of_samples, state_dim)

            scores = ground_truth.get_scores(
                subjective_state=state,
                action_space_to_score=action_space,
                exploit=True,
            )
            # action = torch.argmax(scores, dim=1).unsqueeze(-1)
            # Sample one random action per state from observed actions only
            action = torch.randint(
                low=0, high=unobserved_actions_first_index, size=(number_of_samples, 1)
            )

            # Update action counts
            counter.update(action)

            # Get the base reward from the ground truth model
            base_reward = scores.gather(1, action)

            # Add Gaussian noise to the reward
            noise_scale = 1  # Controls the amount of noise
            noise = torch.randn_like(base_reward) * noise_scale
            noisy_reward = base_reward + noise

            action_representation = ground_truth.action_representation_module(action)

            yield TransitionBatch(
                state=state,
                action=action_representation,
                reward=noisy_reward,
            )

        linear_bandits = [
            LinearBandit(feature_dim=state_dim) for _ in range(number_of_actions)
        ]

        new_policy_learner = DisjointBanditContainer(
            feature_dim=state_dim,
            arm_bandits=linear_bandits,  # pyre-ignore
            exploration_module=DisjointUCBExploration(alpha=0),
            state_features_only=True,
        )

        num_batches = 100
        batch_size = 100

        # List to store MSE values for each batch
        mse_values = []

        for batch_idx in range(num_batches):
            batch = next(
                generate_batch(
                    batch_size,
                    state_dim,
                    ground_truth,
                    training_counter,
                    unobserved_actions_first_index,
                )
            )

            # Calculate predictions before learning
            state = batch.state
            action_indices = batch.action

            # Get predictions from the learned model
            scores = new_policy_learner.get_scores(
                subjective_state=state,
                action_space_to_score=action_space,
                exploit=True,
            )

            # Extract predictions for the actions that were actually taken
            predictions = torch.zeros_like(batch.reward)
            for i, action_idx in enumerate(action_indices):
                predictions[i] = scores[i, action_idx.item()]

            # Calculate MSE between predictions and actual rewards
            mse = torch.mean(torch.pow(predictions - batch.reward, 2)).item()
            mse_values.append(mse)

            # Learn from the batch
            new_policy_learner.learn_batch(batch)

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Processed {batch_idx + 1}/{num_batches} batches, Current MSE: {mse:.6f}"
                )

        # Print training action distribution
        print("\nTraining action distribution:")
        for action_idx in range(number_of_actions):
            count = training_counter.counts[action_idx].item()
            percentage = (count / (num_batches * batch_size)) * 100
            print(f"Action {action_idx}: {count:.0f} occurrences ({percentage:.2f}%)")

        # Assert that no training data was generated with any of the unobserved actions
        for action_idx in range(unobserved_actions_first_index, number_of_actions):
            self.assertEqual(
                training_counter.counts[action_idx].item(),
                0,
                f"Action {action_idx} should never be selected during training, "
                f"but was selected "
                f"{training_counter.counts[action_idx].item()} times",
            )

        # Now check if ground truth and learned model agree on exploit actions
        # for a set of random states
        num_test_states = 1000
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

        # Check that learned scores for unobserved actions are identical across all test examples
        for action_idx in range(unobserved_actions_first_index, number_of_actions):
            unobserved_action_scores = learned_scores[:, action_idx]
            # All scores for this action should be identical (or very close)
            self.assertTrue(
                torch.allclose(
                    unobserved_action_scores,
                    unobserved_action_scores[0].expand_as(unobserved_action_scores),
                    atol=1e-5,
                ),
                f"Learned scores for action {action_idx} should be "
                f"identical for all evaluation examples",
            )

        # Count occurrences of each action in evaluation
        gt_eval_action_counts = torch.zeros(number_of_actions)
        learned_eval_action_counts = torch.zeros(number_of_actions)

        for a in ground_truth_actions:
            gt_eval_action_counts[a.item()] += 1

        for a in learned_actions:
            learned_eval_action_counts[a.item()] += 1

        # Assert that none of the unobserved actions are selected in evaluation
        for action_idx in range(unobserved_actions_first_index, number_of_actions):
            self.assertEqual(
                gt_eval_action_counts[action_idx].item(),
                0,
                f"Action {action_idx} should never be selected in ground truth evaluation, "
                f"but was selected "
                f"{gt_eval_action_counts[action_idx].item()} times",
            )

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

        # Calculate final MSE
        final_mse = mse_values[-1]
        print(f"Final MSE: {final_mse:.6f}")

        # Calculate agreement percentage
        agreement = (ground_truth_actions == learned_actions).float().mean().item()
        agreement_percentage = agreement * 100

        # Given rewards are noise, we don't expect complete agreement
        # The following is based on empirical observations
        required_agreement_percentage = 80.0
        print(f"\nAction agreement percentage: {agreement_percentage:.2f}%")
        error_msg = (
            f"Ground truth and learned model only agree on "
            f"{agreement_percentage:.2f}% of actions"
        )
        self.assertGreaterEqual(
            agreement_percentage, required_agreement_percentage, error_msg
        )
