#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.history_summarization_modules.stacking_history_summarization_module import (
    StackingHistorySummarizationModule,
)


class TestHistorySummarizationModules(unittest.TestCase):
    def setUp(self) -> None:
        self.observation_dim: int = 10
        self.history_length: int = 5

    def test_stacking_history_summarizer(self) -> None:
        """
        Easy test for stacking history summarization module.
        """
        summarization_module = StackingHistorySummarizationModule(
            self.observation_dim, self.history_length
        )
        for _ in range(10):
            observation = torch.rand((1, self.observation_dim))
            subjective_state = summarization_module.summarize_history(observation)
            self.assertEqual(
                subjective_state.shape[0], self.history_length * self.observation_dim
            )
