# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

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
