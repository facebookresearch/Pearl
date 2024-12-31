# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

import torch
import torch.nn as nn
from pearl.api.action import Action
from pearl.api.observation import Observation
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.utils.module_utils import modules_have_similar_state_dict


class LSTMHistorySummarizationModule(HistorySummarizationModule):
    """
    A history summarization module that uses a recurrent neural network
    to summarize past history observations into a hidden representation
    and incrementally generate a new subjective state.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        history_length: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.register_buffer("default_action", torch.zeros((1, action_dim)))
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.action_dim + self.observation_dim)),
        )
        self.lstm = nn.LSTM(
            num_layers=self.num_layers,
            input_size=self.action_dim + self.observation_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

    def summarize_history(
        self, observation: Observation, action: Action | None
    ) -> torch.Tensor:
        assert isinstance(observation, torch.Tensor)
        observation = (
            observation.clone().detach().float().view((1, self.observation_dim))
        )
        if action is None:
            action = self.default_action
        assert isinstance(action, torch.Tensor)
        action = action.clone().detach().float().view((1, self.action_dim))
        observation_action_pair = torch.cat((action, observation.view(1, -1)), dim=-1)

        assert observation.shape[-1] + action.shape[-1] == self.history.shape[-1]
        self.history = torch.cat(
            [
                self.history[1:, :],
                observation_action_pair.view(
                    (1, self.action_dim + self.observation_dim)
                ),
            ],
            dim=0,
        )
        out, (_, _) = self.lstm(self.history)
        return out[-1]

    def get_history(self) -> torch.Tensor:
        return self.history

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (_, _) = self.lstm(x)
        return out[:, -1, :]

    def reset(self) -> None:
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.action_dim + self.observation_dim)),
        )

    def compare(self, other: HistorySummarizationModule) -> str:
        """
        Compares two HistorySummarizationModule instances for equality,
        checking attributes and LSTM state.

        Args:
        other: The other HistorySummarizationModule to compare with.

        Returns:
        str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, LSTMHistorySummarizationModule):
            differences.append(
                "other is not an instance of LSTMHistorySummarizationModule"
            )
        assert isinstance(other, LSTMHistorySummarizationModule)
        if self.history_length != other.history_length:
            differences.append(
                f"history_length is different: {self.history_length} vs {other.history_length}"
            )
        if self.hidden_dim != other.hidden_dim:
            differences.append(
                f"hidden_dim is different: {self.hidden_dim} vs {other.hidden_dim}"
            )
        if self.num_layers != other.num_layers:
            differences.append(
                f"num_layers is different: {self.num_layers} vs {other.num_layers}"
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
        if (reason := modules_have_similar_state_dict(self.lstm, other.lstm)) != "":
            differences.append(f"lstm is different: {reason}")

        return "\n".join(differences)  # Join the differences with newlines
