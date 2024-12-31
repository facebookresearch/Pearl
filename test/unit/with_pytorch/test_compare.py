# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import torch
from later.unittest import TestCase
from pearl.history_summarization_modules.identity_history_summarization_module import (
    IdentityHistorySummarizationModule,
)
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.history_summarization_modules.stacking_history_summarization_module import (
    StackingHistorySummarizationModule,
)
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.neural_networks.contextual_bandit.neural_linear_regression import (
    NeuralLinearRegression,
)


class TestCompare(TestCase):
    """
    A suite of tests for `compare` methods of Pearl classes
    """

    def test_compare_lstm_history_summarization_module(self) -> None:
        module1 = LSTMHistorySummarizationModule(
            history_length=10,
            hidden_dim=32,
            num_layers=2,
            observation_dim=6,
            action_dim=4,
        )
        module2 = LSTMHistorySummarizationModule(
            history_length=10,
            hidden_dim=32,
            num_layers=2,
            observation_dim=6,
            action_dim=4,
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should be different due to random LSTM init)
        self.assertNotEqual(module1.compare(module2), "")

        # Make the LSTMs have the same weights
        for param1, param2 in zip(module1.lstm.parameters(), module2.lstm.parameters()):
            param2.data.copy_(param1.data)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_stacking_history_summarization_module(self) -> None:
        module1 = StackingHistorySummarizationModule(
            observation_dim=6, action_dim=4, history_length=10
        )
        module2 = StackingHistorySummarizationModule(
            observation_dim=6, action_dim=4, history_length=10
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.history_length = 12

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

    def test_compare_identity_history_summarization_module(self) -> None:
        module1 = IdentityHistorySummarizationModule()
        module2 = IdentityHistorySummarizationModule()

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences)
        self.assertEqual(module1.compare(module2), "")

    def test_compare_linear_regression(self) -> None:
        module1 = LinearRegression(feature_dim=10, l2_reg_lambda=0.1, gamma=0.95)
        module2 = LinearRegression(feature_dim=10, l2_reg_lambda=0.1, gamma=0.95)

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have no differences initially)
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.gamma = 0.9

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Make some data and learn it
        x = torch.randn(10, 10)
        y = torch.randn(10, 1)
        module1.learn_batch(x, y)

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Set gamma to the same value as module1's and learn the same data
        module2.gamma = module1.gamma
        module2.learn_batch(x, y)

        # Now the comparison should show no differences
        self.assertEqual(module1.compare(module2), "")

    def test_compare_neural_linear_regression(self) -> None:
        module1 = NeuralLinearRegression(
            feature_dim=10, hidden_dims=[32, 16], l2_reg_lambda_linear=0.1, gamma=0.95
        )
        module2 = NeuralLinearRegression(
            feature_dim=10, hidden_dims=[32, 16], l2_reg_lambda_linear=0.1, gamma=0.95
        )

        # Compare module1 with itself
        self.assertEqual(module1.compare(module1), "")

        # Compare module1 with module2 (should have random differences initially)
        self.assertNotEqual(module1.compare(module2), "")

        # Load state dict from one to the other and compare again
        module1.load_state_dict(module2.state_dict())
        self.assertEqual(module1.compare(module2), "")

        # Modify an attribute of module2 to create a difference
        module2.nn_e2e = not module2.nn_e2e

        # Now the comparison should show a difference
        self.assertNotEqual(module1.compare(module2), "")

        # Undo flip, load state dict from one to the other and compare again
        module2.nn_e2e = not module2.nn_e2e
        module1.load_state_dict(module2.state_dict())
        self.assertEqual(module1.compare(module2), "")
