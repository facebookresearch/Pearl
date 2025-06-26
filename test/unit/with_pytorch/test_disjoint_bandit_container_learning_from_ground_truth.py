# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest
from typing import List, Tuple

import torch
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
)

from pearl.policy_learners.contextual_bandits.disjoint_bandit import (
    DisjointBanditContainer,
)
from pearl.policy_learners.contextual_bandits.linear_bandit import LinearBandit
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    DisjointUCBExploration,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

# Helper functions for disjoint bandit testing


class ActionCounter:
    """Class to hold action counts to avoid nonlocal issues."""

    def __init__(self, num_actions: int) -> None:
        self.counts: torch.Tensor = torch.zeros(num_actions)

    def update(self, actions: torch.Tensor) -> None:
        for a in actions:
            self.counts[a.item()] += 1


def create_ground_truth_model(
    state_dim: int, number_of_actions: int, unobserved_actions_first_index: int
) -> Tuple[DisjointBanditContainer, DiscreteActionSpace]:
    """
    Create a ground truth model with linear bandits for each action.

    Args:
        state_dim: Dimension of the state space
        number_of_actions: Number of actions in the action space
        unobserved_actions_first_index: Actions >= this index should never be selected

    Returns:
        Tuple of (ground_truth_model, action_space)
    """
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

    return ground_truth, action_space


def generate_batch(
    number_of_samples: int,
    state_dim: int,
    ground_truth: DisjointBanditContainer,
    counter: ActionCounter,
    unobserved_actions_first_index: int,
    action_space: DiscreteActionSpace,
    noise_scale: float = 0.1,
) -> TransitionBatch:
    """
    Generate a batch of transitions using the ground truth model.

    Args:
        number_of_samples: Number of samples to generate
        state_dim: Dimension of the state space
        ground_truth: Ground truth model
        counter: Counter to track action occurrences
        unobserved_actions_first_index: Actions >= this index should never be selected
        action_space: Action space
        noise_scale: Scale of Gaussian noise to add to rewards

    Returns:
        A TransitionBatch
    """
    # Generate random state
    state = torch.rand(number_of_samples, state_dim)

    scores = ground_truth.get_scores(
        subjective_state=state,
        action_space_to_score=action_space,
        exploit=True,
    )

    # Sample one random action per state from observed actions only
    action = torch.randint(
        low=0, high=unobserved_actions_first_index, size=(number_of_samples, 1)
    )

    # Update action counts
    counter.update(action)

    # Get the base reward from the ground truth model
    base_reward = scores.gather(1, action)

    # Add Gaussian noise to the reward
    noise = torch.randn_like(base_reward) * noise_scale
    noisy_reward = base_reward + noise

    action_representation = ground_truth.action_representation_module(action)

    return TransitionBatch(
        state=state,
        action=action_representation,
        reward=noisy_reward,
    )


def create_policy_learner(
    state_dim: int, number_of_actions: int
) -> DisjointBanditContainer:
    """
    Create a new policy learner with linear bandits for each action.

    Args:
        state_dim: Dimension of the state space
        number_of_actions: Number of actions in the action space

    Returns:
        A DisjointBanditContainer policy learner
    """
    linear_bandits: List[ContextualBanditBase] = [
        LinearBandit(feature_dim=state_dim) for _ in range(number_of_actions)
    ]

    return DisjointBanditContainer(
        feature_dim=state_dim,
        arm_bandits=linear_bandits,
        exploration_module=DisjointUCBExploration(alpha=0),
        state_features_only=True,
    )


def train_model(
    policy_learner: DisjointBanditContainer,
    ground_truth: DisjointBanditContainer,
    state_dim: int,
    action_space: DiscreteActionSpace,
    training_counter: ActionCounter,
    unobserved_actions_first_index: int,
    num_batches: int,
    batch_size: int,
    noise_scale: float = 0.1,
) -> List[float]:
    """
    Train the policy learner on batches generated from the ground truth model.

    Args:
        policy_learner: Policy learner to train
        ground_truth: Ground truth model
        state_dim: Dimension of the state space
        action_space: Action space
        training_counter: Counter to track action occurrences
        unobserved_actions_first_index: Actions >= this index should never be selected
        num_batches: Number of batches to train on
        batch_size: Size of each batch
        noise_scale: Scale of Gaussian noise to add to rewards

    Returns:
        List of MSE values for each batch
    """
    # List to store MSE values for each batch
    mse_values = []

    for batch_idx in range(num_batches):
        batch = generate_batch(
            batch_size,
            state_dim,
            ground_truth,
            training_counter,
            unobserved_actions_first_index,
            action_space,
            noise_scale,
        )

        # Calculate predictions before learning
        state = batch.state
        action_indices = batch.action

        # Get predictions from the learned model
        scores = policy_learner.get_scores(
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
        policy_learner.learn_batch(batch)

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed {batch_idx + 1}/{num_batches} batches, Current MSE: {mse:.6f}"
            )

    return mse_values


def evaluate_model(
    policy_learner: DisjointBanditContainer,
    ground_truth: DisjointBanditContainer,
    state_dim: int,
    action_space: DiscreteActionSpace,
    number_of_actions: int,
    unobserved_actions_first_index: int,
    num_test_states: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Evaluate the policy learner against the ground truth model.

    Args:
        policy_learner: Policy learner to evaluate
        ground_truth: Ground truth model
        state_dim: Dimension of the state space
        action_space: Action space
        number_of_actions: Number of actions in the action space
        unobserved_actions_first_index: Actions >= this index should never be selected
        num_test_states: Number of test states to evaluate on

    Returns:
        Tuple of (ground_truth_actions, learned_actions, ground_truth_eval_action_counts,
                 learned_eval_action_counts, agreement_percentage)
    """
    # Generate test states
    test_states = torch.rand(num_test_states, state_dim)

    # Get exploit actions from ground truth model
    ground_truth_scores = ground_truth.get_scores(
        subjective_state=test_states,
        action_space_to_score=action_space,
        exploit=True,
    )
    ground_truth_actions = torch.argmax(ground_truth_scores, dim=1)

    # Get exploit actions from learned model
    learned_scores = policy_learner.get_scores(
        subjective_state=test_states,
        action_space_to_score=action_space,
        exploit=True,
    )
    learned_actions = torch.argmax(learned_scores, dim=1)

    # Check that learned scores for unobserved actions are identical across all test examples
    for action_idx in range(unobserved_actions_first_index, number_of_actions):
        unobserved_action_scores = learned_scores[:, action_idx]
        # All scores for this action should be identical (or very close)
        assert torch.allclose(
            unobserved_action_scores,
            unobserved_action_scores[0].expand_as(unobserved_action_scores),
            atol=1e-5,
        ), f"Learned scores for action {action_idx} should be identical for all evaluation examples"

    # Count occurrences of each action in evaluation
    ground_truth_eval_action_counts = torch.zeros(number_of_actions)
    learned_eval_action_counts = torch.zeros(number_of_actions)

    for a in ground_truth_actions:
        ground_truth_eval_action_counts[a.item()] += 1

    for a in learned_actions:
        learned_eval_action_counts[a.item()] += 1

    # Calculate agreement percentage
    agreement = (ground_truth_actions == learned_actions).float().mean().item()
    agreement_percentage = agreement * 100

    return (
        ground_truth_actions,
        learned_actions,
        ground_truth_eval_action_counts,
        learned_eval_action_counts,
        agreement_percentage,
    )


def print_action_distribution(
    counter: ActionCounter, total_samples: int, title: str = "Action distribution:"
) -> None:
    """
    Print the distribution of actions.

    Args:
        counter: Counter with action counts
        total_samples: Total number of samples
        title: Title for the distribution printout
    """
    print(f"\n{title}")
    for action_idx, count in enumerate(counter.counts):
        count_val = count.item()
        percentage = (count_val / total_samples) * 100
        print(f"Action {action_idx}: {count_val:.0f} occurrences ({percentage:.2f}%)")


def print_evaluation_comparison(
    ground_truth_eval_action_counts: torch.Tensor,
    learned_eval_action_counts: torch.Tensor,
    num_test_states: int,
) -> None:
    """
    Print a comparison of ground truth and learned model action distributions.

    Args:
        ground_truth_eval_action_counts: Counts of actions selected by ground truth model
        learned_eval_action_counts: Counts of actions selected by learned model
        num_test_states: Number of test states
    """
    print("\nEvaluation action distribution:")
    print("Action | Ground Truth | Learned Model")
    print("-----------------------------------")
    for action_idx in range(len(ground_truth_eval_action_counts)):
        gt_count = ground_truth_eval_action_counts[action_idx].item()
        learned_count = learned_eval_action_counts[action_idx].item()
        gt_percentage = (gt_count / num_test_states) * 100
        learned_percentage = (learned_count / num_test_states) * 100
        print(
            f"  {action_idx:<2}  |    {gt_count:>3.0f} ({gt_percentage:>5.1f}%) | "
            f"   {learned_count:>3.0f} ({learned_percentage:>5.1f}%)"
        )


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
        # Setup parameters
        state_dim = 78
        number_of_actions = 24
        unobserved_actions_first_index = (
            16  # Actions >= this index should never be selected
        )
        num_batches = 100
        batch_size = 100
        num_test_states = 1000
        noise_scale = 1.0

        # Create ground truth model and action space
        ground_truth, action_space = create_ground_truth_model(
            state_dim, number_of_actions, unobserved_actions_first_index
        )

        # Initialize counter for action occurrences during training
        training_counter = ActionCounter(number_of_actions)

        # Create policy learner
        policy_learner = create_policy_learner(state_dim, number_of_actions)

        # Train the model and collect MSE values
        mse_values = train_model(
            policy_learner=policy_learner,
            ground_truth=ground_truth,
            state_dim=state_dim,
            action_space=action_space,
            training_counter=training_counter,
            unobserved_actions_first_index=unobserved_actions_first_index,
            num_batches=num_batches,
            batch_size=batch_size,
            noise_scale=noise_scale,
        )

        # Print training action distribution
        print_action_distribution(
            training_counter, num_batches * batch_size, "Training action distribution:"
        )

        # Assert that no training data was generated with any of the unobserved actions
        for action_idx in range(unobserved_actions_first_index, number_of_actions):
            self.assertEqual(
                training_counter.counts[action_idx].item(),
                0,
                f"Action {action_idx} should never be selected during training, "
                f"but was selected "
                f"{training_counter.counts[action_idx].item()} times",
            )

        # Evaluate the model
        (
            ground_truth_actions,
            learned_actions,
            ground_truth_eval_action_counts,
            learned_eval_action_counts,
            agreement_percentage,
        ) = evaluate_model(
            policy_learner=policy_learner,
            ground_truth=ground_truth,
            state_dim=state_dim,
            action_space=action_space,
            number_of_actions=number_of_actions,
            unobserved_actions_first_index=unobserved_actions_first_index,
            num_test_states=num_test_states,
        )

        # Assert that none of the unobserved actions are selected in evaluation
        for action_idx in range(unobserved_actions_first_index, number_of_actions):
            self.assertEqual(
                ground_truth_eval_action_counts[action_idx].item(),
                0,
                f"Action {action_idx} should never be selected in ground truth evaluation, "
                f"but was selected "
                f"{ground_truth_eval_action_counts[action_idx].item()} times",
            )

        # Print evaluation comparison
        print_evaluation_comparison(
            ground_truth_eval_action_counts, learned_eval_action_counts, num_test_states
        )

        # Calculate final MSE
        final_mse = mse_values[-1]
        print(f"Final MSE: {final_mse:.6f}")

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
