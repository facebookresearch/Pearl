# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

import torch
from pearl.api.action import Action
from pearl.api.history import History
from pearl.api.observation import Observation
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.utils.tensor_like import assert_is_tensor_like


class StackingHistorySummarizationModule(HistorySummarizationModule):
    """
    A history summarization module that simply stacks observations into a history.
    """

    def __init__(
        self, observation_dim: int, action_dim: int, history_length: int = 8
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.register_buffer("default_action", torch.zeros((1, action_dim)))
        self.register_buffer(
            "history", torch.zeros((history_length, action_dim + observation_dim))
        )

    def summarize_history(
        self, observation: Observation, action: Action | None
    ) -> torch.Tensor:
        if action is None:
            action = self.default_action

        observation = assert_is_tensor_like(observation)
        action = assert_is_tensor_like(action)
        assert observation.shape[-1] + action.shape[-1] == self.history.shape[-1]
        observation_action_pair = torch.cat(
            (action, observation.view(1, -1)), dim=-1
        ).detach()
        self.history = torch.cat(
            [
                self.history[1:, :],
                observation_action_pair.view(
                    (1, self.action_dim + self.observation_dim)
                ),
            ],
            dim=0,
        )
        return self.history.view(-1)

    def get_history(self) -> torch.Tensor:
        return self.history.view(-1)

    def forward(self, x: History) -> torch.Tensor:
        x = assert_is_tensor_like(x)
        return x

    def reset(self) -> None:
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.action_dim + self.observation_dim)),
        )

    def compare(self, other: HistorySummarizationModule) -> str:
        """
        Compares two StackingHistorySummarizationModule instances for equality,
        checking attributes and history buffer.

        Args:
        other: The other StackingHistorySummarizationModule to compare with.

        Returns:
        str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, StackingHistorySummarizationModule):
            differences.append(
                "other is not an instance of StackingHistorySummarizationModule"
            )
        assert isinstance(other, StackingHistorySummarizationModule)
        if self.history_length != other.history_length:
            differences.append(
                f"history_length is different: {self.history_length} vs {other.history_length}"
            )
        if self.observation_dim != other.observation_dim:
            differences.append(
                f"observation_dim is different: {self.observation_dim} vs {other.observation_dim}"
            )
        if self.action_dim != other.action_dim:
            differences.append(
                f"action_dim is different: {self.action_dim} vs {other.action_dim}"
            )
        if not torch.allclose(self.default_action, other.default_action):
            differences.append(
                f"default_action is different: {self.default_action} vs {other.default_action}"
            )
        if not torch.allclose(self.history, other.history):
            differences.append(
                f"history is different: {self.history} vs {other.history}"
            )

        return "\n".join(differences)  # Join the differences with newlines
