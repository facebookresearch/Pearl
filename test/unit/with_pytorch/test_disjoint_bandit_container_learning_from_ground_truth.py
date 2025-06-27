# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import os
import unittest
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import torch

from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
)
from pearl.policy_learners.contextual_bandits.disjoint_bandit import (
    DisjointBanditContainer,
)
from pearl.policy_learners.contextual_bandits.linear_bandit import LinearBandit
from pearl.policy_learners.exploration_modules.common.tiebreaking_strategy import (
    TiebreakingStrategy,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    DisjointUCBExploration,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

# Helper functions for disjoint bandit testing


def create_ground_truth_model(
    state_dim: int, number_of_actions: int, zeroed_actions_first_index: int = -1
) -> Tuple[DisjointBanditContainer, DiscreteActionSpace]:
    """
    Create a ground truth model with linear bandits for each action.

    Args:
        state_dim: Dimension of the state space
        number_of_actions: Number of actions in the action space
        zeroed_actions_first_index: Actions >= this index will have zeroed coefficients
                                    Defaults to number_of_actions, meaning no actions are zeroed

    Returns:
        Tuple of (ground_truth_model, action_space)
    """
    if zeroed_actions_first_index < 0:
        zeroed_actions_first_index = number_of_actions

    action_space: DiscreteActionSpace = DiscreteActionSpace(
        [torch.tensor([i]) for i in range(number_of_actions)]
    )

    ground_truth_linear_bandits = []
    for action_idx in range(number_of_actions):
        true_coefs = torch.zeros(state_dim + 1)  # +1 for intercept

        if action_idx >= zeroed_actions_first_index:
            true_coefs = torch.zeros(state_dim + 1)
        else:
            true_coefs[0] = 0  # intercept
            true_coefs[1:] = torch.rand(state_dim) * 2 - 1

        ground_truth_linear_bandit = LinearBandit(
            feature_dim=state_dim,
            initial_coefs=true_coefs,
        )
        ground_truth_linear_bandits.append(ground_truth_linear_bandit)

    ground_truth = DisjointBanditContainer(
        feature_dim=state_dim,
        arm_bandits=ground_truth_linear_bandits,
        exploration_module=DisjointUCBExploration(
            alpha=0, randomized_tiebreaking=TiebreakingStrategy.PER_ROW_TIEBREAKING
        ),
        state_features_only=True,
    )

    return ground_truth, action_space


def generate_batch(
    training_round: int,
    model: DisjointBanditContainer,
    number_of_samples: int,
    state_dim: int,
    ground_truth: DisjointBanditContainer,
    unobserved_actions_first_index: int,
    action_space: DiscreteActionSpace,
    noise_scale: float = 0.1,
) -> Tuple[TransitionBatch, torch.Tensor]:
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

    if training_round == 0:
        # Sample one random action per state from observed actions only.
        # We do this to verify that model coefficients for unobserved actions
        # remain unchanged.
        action = torch.randint(
            low=0, high=unobserved_actions_first_index, size=(number_of_samples, 1)
        )
    else:
        # After first training round, we let the learned model to select actions
        action = model.act(state, action_space, exploit=True)

    # Count actions
    action_counts = torch.zeros(action_space.n)
    for a in action:
        action_counts[a.item()] += 1

    # Get the base reward from the ground truth model
    base_reward = scores.gather(1, action)

    # Add Gaussian noise to the reward
    noise = torch.randn_like(base_reward) * noise_scale
    noisy_reward = base_reward + noise

    action_representation = ground_truth.action_representation_module(action)

    return (
        TransitionBatch(
            state=state,
            action=action_representation,
            reward=noisy_reward,
        ),
        action_counts,
    )


def create_model(state_dim: int, number_of_actions: int) -> DisjointBanditContainer:
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
        exploration_module=DisjointUCBExploration(
            alpha=1, randomized_tiebreaking=TiebreakingStrategy.PER_ROW_TIEBREAKING
        ),
        state_features_only=True,
    )


def train_model(
    training_round: int,
    model: DisjointBanditContainer,
    ground_truth: DisjointBanditContainer,
    state_dim: int,
    action_space: DiscreteActionSpace,
    unobserved_actions_first_index: int,
    num_batches: int,
    batch_size: int,
    noise_scale: float = 0.1,
) -> Tuple[List[float], torch.Tensor]:
    """
    Train the policy learner on batches generated from the ground truth model.

    Args:
        model: Policy learner to train
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
    # List to store MSE values for each batch and initialize action counts
    mse_values = []
    training_action_counts = torch.zeros(action_space.n)

    for batch_idx in range(num_batches):
        batch, batch_action_counts = generate_batch(
            training_round,
            model,
            batch_size,
            state_dim,
            ground_truth,
            unobserved_actions_first_index,
            action_space,
            noise_scale,
        )

        # Update action counts
        training_action_counts += batch_action_counts

        # Calculate predictions before learning
        state = batch.state
        action_indices = batch.action

        # Get predictions from the learned model
        scores = model.get_scores(
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
        model.learn_batch(batch)

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed {batch_idx + 1}/{num_batches} batches, Current MSE: {mse:.6f}"
            )

    return mse_values, training_action_counts


def evaluate_model(
    training_round: int,
    model: DisjointBanditContainer,
    ground_truth: DisjointBanditContainer,
    state_dim: int,
    action_space: DiscreteActionSpace,
    number_of_actions: int,
    unobserved_actions_first_index: int,
    num_test_states: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Evaluate the policy learner against the ground truth model.

    Args:
        model: Policy learner to evaluate
        ground_truth: Ground truth model
        state_dim: Dimension of the state space
        action_space: Action space
        number_of_actions: Number of actions in the action space
        unobserved_actions_first_index: Actions >= this index should never be selected
        num_test_states: Number of test states to evaluate on

    Returns:
        Tuple of (ground_truth_eval_action_counts,
                 learned_eval_action_counts, agreement_percentage)
    """
    # Generate test states
    test_states = torch.rand(num_test_states, state_dim)

    # Get exploit actions from ground truth model
    ground_truth_actions = ground_truth.act(
        subjective_state=test_states,
        available_action_space=action_space,
        exploit=True,
    )

    # Check that learned exploit scores for unobserved actions
    # are identical across all test examples in the first round
    if training_round == 0:
        learned_scores = model.get_scores(
            subjective_state=test_states,
            action_space_to_score=action_space,
            exploit=True,
        )
        for action_idx in range(unobserved_actions_first_index, number_of_actions):
            unobserved_action_scores = learned_scores[:, action_idx]
            # All scores for this action should be identical (or very close)
            assert torch.allclose(
                unobserved_action_scores,
                unobserved_action_scores[0].expand_as(unobserved_action_scores),
                atol=1e-5,
            ), (
                f"Learned scores for action {action_idx} should be "
                "identical for all evaluation examples"
            )

    learned_actions = model.act(
        subjective_state=test_states,
        available_action_space=action_space,
        exploit=True,
    )

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
        ground_truth_eval_action_counts,
        learned_eval_action_counts,
        agreement_percentage,
    )


def plot_mse(mse_values: List[float], window_size: int = 5) -> str:
    """
    Plot MSE values and save to a file.

    Args:
        mse_values: List of MSE values
        window_size: Window size for moving average

    Returns:
        Path to the saved plot
    """
    num_batches = len(mse_values)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_batches + 1), mse_values, "b-")
    plt.title("MSE between Batch Rewards and Predictions vs. Batch Number")
    plt.xlabel("Batch Number")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)

    # Add a smoothed curve using moving average
    if num_batches >= window_size:
        smoothed_mse = np.convolve(
            mse_values, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            range(window_size, num_batches + 1),
            smoothed_mse,
            "r-",
            label=f"Moving Average (window={window_size})",
        )
        plt.legend()

    # Save the plot
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, "disjoint_bandit_mse.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def print_action_distribution(
    action_counts: torch.Tensor, total_samples: int, title: str = "Action distribution:"
) -> None:
    """
    Print the distribution of actions.

    Args:
        action_counts: Tensor with action counts
        total_samples: Total number of samples
        title: Title for the distribution printout
    """
    print(f"\n{title}")
    print("Action |  Count (Percentage)  | Histogram")
    print("-------------------------------------------")
    for action_idx, count in enumerate(action_counts):
        count_val = count.item()
        percentage = (count_val / total_samples) * 100

        # Format with consistent spacing for better alignment
        action_str = f"   {action_idx:<2}"
        count_str = f"{count_val:>5.0f} ({percentage:>6.2f}%)"

        # Create histogram bar (0-15 blocks based on percentage)
        # Round to nearest block, max 15
        num_blocks = min(15, int(percentage * 15 / 100 + 0.5))
        histogram = "█" * num_blocks

        print(f"{action_str}  | {count_str:>20} | {histogram}")


def print_evaluation_comparison(
    training_round: int,
    number_of_training_rounds: int,
    ground_truth_eval_action_counts: torch.Tensor,
    learned_eval_action_counts: torch.Tensor,
    num_test_states: int,
    unobserved_actions_first_index: int,
) -> None:
    """
    Print a comparison of ground truth and learned model action distributions.

    Args:
        ground_truth_eval_action_counts: Counts of actions selected by ground truth model
        learned_eval_action_counts: Counts of actions selected by learned model
        num_test_states: Number of test states
        unobserved_actions_first_index: Actions >= this index should never be selected
    """
    print(
        f"\nEvaluation action distribution after training round "
        f"{training_round + 1}/{number_of_training_rounds}:"
    )
    print("Action |  Ground Truth  |  Histogram   | Learned Model  | Histogram")
    print("-------------------------------------------------------------------")
    for action_idx in range(len(ground_truth_eval_action_counts)):
        gt_count = ground_truth_eval_action_counts[action_idx].item()
        learned_count = learned_eval_action_counts[action_idx].item()
        gt_percentage = (gt_count / num_test_states) * 100
        learned_percentage = (learned_count / num_test_states) * 100

        # Format with consistent spacing for better alignment
        action_str = f"   {action_idx:<2}"
        gt_str = f"{gt_count:>5.0f} ({gt_percentage:>5.1f}%)"
        learned_str = f"{learned_count:>5.0f} ({learned_percentage:>5.1f}%)"

        # Create histogram bars (0-15 blocks based on percentage)
        # Round to nearest block, max 15
        gt_blocks = min(15, int(gt_percentage * 15 / 100 + 0.5))
        learned_blocks = min(15, int(learned_percentage * 15 / 100 + 0.5))
        gt_histogram = "█" * gt_blocks
        learned_histogram = "█" * learned_blocks

        # Print the row with histograms
        # Split into two print statements to avoid line length issues
        print(f"{action_str}  | {gt_str:>13} | {gt_histogram:12}", end="")
        print(f" | {learned_str:>13} | {learned_histogram:12}", end="")

        if training_round == 0 and action_idx == unobserved_actions_first_index:
            print(" (first non-observed action)")
        else:
            print()


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
        mini = False
        if mini:
            number_of_training_rounds = 25
            number_of_actions = 8
            unobserved_actions_first_index = 4  # Actions >= this not in initial data
            state_dim = 6
            num_batches = 10
            batch_size = 2
            num_test_states = 1_000
            noise_scale = 0.1
            required_agreement_percentage = 80.0  # empirical safe lower bound
        else:
            number_of_training_rounds = 8
            number_of_actions = 24
            unobserved_actions_first_index = 5  # Actions >= this not in initial data
            state_dim = 78
            num_batches = 50
            batch_size = 100
            num_test_states = 10_000
            noise_scale = 1.0
            required_agreement_percentage = 80.0  # empirical safe lower bound

        # Create ground truth model and action space
        # We zero out the coefficients for actions >= unobserved_actions_first_index
        # to match the learned model, which will always be zero and never updated
        # because their actions are never selected during training.
        # We do this so the agreement test down below makes sense.
        ground_truth, action_space = create_ground_truth_model(
            state_dim,
            number_of_actions,
            zeroed_actions_first_index=number_of_actions,
        )

        # Create model
        model = create_model(state_dim, number_of_actions)

        for training_round in range(number_of_training_rounds):
            print(f"\nRound {training_round + 1}/{number_of_training_rounds}:")

            # Train the model and collect MSE values and action counts
            mse_values, training_action_counts = train_model(
                training_round,
                model=model,
                ground_truth=ground_truth,
                state_dim=state_dim,
                action_space=action_space,
                unobserved_actions_first_index=unobserved_actions_first_index,
                num_batches=num_batches,
                batch_size=batch_size,
                noise_scale=noise_scale,
            )

            # Print training action distribution
            print_action_distribution(
                training_action_counts,
                num_batches * batch_size,
                "Training action distribution after "
                + f"training round {training_round + 1}/{number_of_training_rounds}:",
            )
            if training_round == 0:
                print(
                    f"In the first round we have access to a database with only "
                    f"actions up to {unobserved_actions_first_index - 1}."
                )
                print(
                    "In the following rounds, we use the model being learned "
                    "to act on training data according to an exploration policy."
                )

                # Assert that no training data was generated with any of the unobserved actions.
                # We do this in training round 0 only because after that
                # the learned model is allowed to explore.
                for action_idx in range(
                    unobserved_actions_first_index, number_of_actions
                ):
                    self.assertEqual(
                        training_action_counts[action_idx].item(),
                        0,
                        f"Action {action_idx} should never be selected during training, "
                        f"but was selected "
                        f"{training_action_counts[action_idx].item()} times",
                    )

            # Evaluate the model
            (
                ground_truth_eval_action_counts,
                learned_eval_action_counts,
                agreement_percentage,
            ) = evaluate_model(
                training_round=training_round,
                model=model,
                ground_truth=ground_truth,
                state_dim=state_dim,
                action_space=action_space,
                number_of_actions=number_of_actions,
                unobserved_actions_first_index=unobserved_actions_first_index,
                num_test_states=num_test_states,
            )

            # Print evaluation comparison
            print_evaluation_comparison(
                training_round,
                number_of_training_rounds,
                ground_truth_eval_action_counts,
                learned_eval_action_counts,
                num_test_states,
                unobserved_actions_first_index,
            )

            # Plot MSE values
            plot_path = plot_mse(mse_values, window_size=5)
            print(f"\nMSE plot saved to: {plot_path}")

            # Calculate final MSE
            final_mse = mse_values[-1]
            print(f"Final MSE: {final_mse:.6f}")

            # Given rewards are noise, we don't expect complete agreement
            # The following is based on empirical observations
            print(f"\nAction agreement percentage: {agreement_percentage:.2f}%")
            if training_round == number_of_training_rounds - 1:
                self.assertGreaterEqual(
                    agreement_percentage,
                    required_agreement_percentage,
                    f"Ground truth and learned model only agree on "
                    f"{agreement_percentage:.2f}% of actions",
                )
