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

from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_append_ones(self) -> None:
        # 1D input
        x = torch.randn(10)
        expected_output = torch.cat([torch.ones((1,)), x], dim=0)
        output = LinearRegression.append_ones(x)
        tt.assert_close(expected_output, output)

        # 2D input
        x = torch.randn(10, 5)
        expected_output = torch.cat([torch.ones((10, 1)), x], dim=1)
        output = LinearRegression.append_ones(x)
        tt.assert_close(expected_output, output)

        # 3D input
        x = torch.randn(10, 5, 6)
        expected_output = torch.cat([torch.ones((10, 5, 1)), x], dim=2)
        output = LinearRegression.append_ones(x)
        tt.assert_close(expected_output, output)

        # make sure it's traceable
        _ = torch.fx.symbolic_trace(LinearRegression.append_ones)

    def test_linear_regression_random(self) -> None:
        feature_dim: int = 15
        batch_size: int = feature_dim * 4
        # it is important to have enough data for training

        linear_regression = LinearRegression(feature_dim=feature_dim)
        losses = []
        feature = torch.randn(batch_size, feature_dim)
        reward = feature.sum(-1, keepdim=True)
        weight = torch.ones(batch_size, 1)
        for _ in range(10):
            loss = (linear_regression(feature) - reward) ** 2
            losses.append(loss.mean().item())
            linear_regression.learn_batch(x=feature, y=reward, weight=weight)
        losses.append(loss.mean().item())

        self.assertGreater(sum(losses[:5]), sum(losses[-5:]))
        self.assertGreater(1e-2, losses[-1])

    def test_state_dict(self) -> None:
        feature_dim = 15
        model = LinearRegression(feature_dim=feature_dim)
        states = model.state_dict()
        self.assertEqual(len(states), 5)
        self.assertEqual(states["_A"].shape, (feature_dim + 1, feature_dim + 1))
        self.assertEqual(states["_b"].shape, (feature_dim + 1,))
        self.assertEqual(states["_inv_A"].shape, (feature_dim + 1, feature_dim + 1))
        self.assertEqual(states["_coefs"].shape, (feature_dim + 1,))
        self.assertEqual(states["_sum_weight"].shape, (1,))
        states["_b"] = torch.ones((feature_dim + 1,))
        model.load_state_dict(states)
        self.assertEqual(model._b[3], 1)

    def test_coefficient_recovery(self) -> None:
        """
        Test that LinearRegression can recover the ground truth coefficients.
        We create a ground truth coefficient vector, generate data from it,
        train the model, and compare the learned coefficients with the ground truth.
        """
        # Set up ground truth
        feature_dim = 10
        # Create ground truth coefficients (including intercept)
        true_coefs = torch.tensor([2.5] + [0.5 * i for i in range(1, feature_dim + 1)])

        # Create a "ground truth" model initialized with the true coefficients
        ground_truth_model = LinearRegression(
            feature_dim=feature_dim, initial_coefs=true_coefs
        )

        # Initialize model to be trained
        model = LinearRegression(feature_dim=feature_dim)

        # Generate synthetic data
        num_batches = 20
        batch_size = 50

        for _ in range(num_batches):
            # Generate random features
            features = torch.randn(batch_size, feature_dim)

            # Compute target values using the ground truth model
            targets = ground_truth_model(features)

            # Add some noise to make it more realistic
            noise = torch.randn_like(targets) * 0.1
            targets = targets + noise

            # Train the model on this batch
            model.learn_batch(x=features, y=targets)

        # Compare learned coefficients with ground truth
        learned_coefs = model.coefs

        # Check if the learned coefficients are close to the ground truth
        # We use a relatively large tolerance because of the added noise
        tt.assert_close(learned_coefs, true_coefs, rtol=0.1, atol=0.1)

        # Also check that the model predictions are accurate
        test_features = torch.randn(100, feature_dim)
        true_predictions = ground_truth_model(test_features)
        model_predictions = model(test_features)

        # Check prediction accuracy
        tt.assert_close(model_predictions, true_predictions, rtol=0.2, atol=0.2)

    def test_zero_features_zero_coefficients(self) -> None:
        """
        Test that when certain features are always zero, their corresponding
        coefficients in the LinearRegression model also remain zero.

        This test verifies that the zero-coefficient property holds after each
        learn_batch operation, not just at the end of training.
        """
        # Set up ground truth
        feature_dim = 10

        # Create ground truth coefficients where some are non-zero and others are zero
        # We'll make features at indices 2, 5, and 8 have zero coefficients
        zero_feature_indices = [2, 5, 8]
        true_coefs = torch.zeros(feature_dim + 1)  # +1 for intercept

        # Set non-zero coefficients
        intercept = 3.0
        true_coefs[0] = intercept  # intercept
        for i in range(1, feature_dim + 1):
            if (
                i - 1 not in zero_feature_indices
            ):  # -1 because true_coefs[0] is the intercept
                true_coefs[i] = 0.5 * i

        # Create a "ground truth" model initialized with the true coefficients
        ground_truth_model = LinearRegression(
            feature_dim=feature_dim, initial_coefs=true_coefs
        )

        # Initialize model to be trained
        model = LinearRegression(feature_dim=feature_dim)

        # Generate synthetic data
        num_batches = 30  # More batches for better convergence
        batch_size = 50

        # Define the tolerance for zero coefficients
        # Start with a larger tolerance and decrease it as training progresses
        # This accounts for the fact that early in training, coefficients may not be exactly zero
        initial_delta = 0.2
        final_delta = 0.05

        for batch_idx in range(num_batches):
            # Generate random features
            features = torch.randn(batch_size, feature_dim)

            # Set the features at zero_feature_indices to zero
            for idx in zero_feature_indices:
                features[:, idx] = 0.0

            # Compute target values using the ground truth model
            targets = ground_truth_model(features)

            # Add some noise to make it more realistic
            noise = (
                torch.randn_like(targets) * 0.05
            )  # Less noise for better convergence
            targets = targets + noise

            # Train the model on this batch
            model.learn_batch(x=features, y=targets)

            # Calculate the current tolerance based on training progress
            # Linearly decrease from initial_delta to final_delta
            current_delta = initial_delta - (initial_delta - final_delta) * (
                batch_idx / (num_batches - 1)
            )

            # After each batch, verify that coefficients for zero features remain zero
            for idx in zero_feature_indices:
                # +1 because learned_coefs[0] is the intercept
                coef_value = model.coefs[idx + 1].item()
                self.assertAlmostEqual(
                    coef_value,
                    0.0,
                    delta=current_delta,
                    msg=(
                        f"After batch {batch_idx}, coefficient for zero feature {idx} "
                        f"is {coef_value}, which exceeds tolerance {current_delta}"
                    ),
                )

        # Compare final learned coefficients with ground truth
        learned_coefs = model.coefs

        # Check if the learned coefficients are close to the ground truth
        tt.assert_close(learned_coefs, true_coefs, rtol=0.1, atol=0.1)

        # Specifically check that the final coefficients for zero features are very close to zero
        for idx in zero_feature_indices:
            # +1 because learned_coefs[0] is the intercept
            self.assertAlmostEqual(
                learned_coefs[idx + 1].item(),
                0.0,
                delta=final_delta,
                msg=(
                    f"Final coefficient for zero feature {idx} is {learned_coefs[idx + 1].item()}, "
                    f"which exceeds tolerance {final_delta}"
                ),
            )

        # Also check that the model predictions are accurate
        test_features = torch.randn(100, feature_dim)
        # Set the features at zero_feature_indices to zero
        for idx in zero_feature_indices:
            test_features[:, idx] = 0.0

        true_predictions = ground_truth_model(test_features)
        model_predictions = model(test_features)

        # Check prediction accuracy
        tt.assert_close(model_predictions, true_predictions, rtol=0.2, atol=0.2)
