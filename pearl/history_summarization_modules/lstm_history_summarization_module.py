from typing import Optional

import torch
import torch.nn as nn
from pearl.api.action import Action
from pearl.api.observation import Observation
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)


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
        state_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super(LSTMHistorySummarizationModule, self).__init__()
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
        self.register_buffer(
            "cell_representation", torch.zeros((num_layers, hidden_dim))
        )
        self.register_buffer(
            "hidden_representation", torch.zeros((num_layers, hidden_dim))
        )
        self.register_buffer(
            "default_cell_representation", torch.zeros((num_layers, hidden_dim))
        )
        self.register_buffer(
            "default_hidden_representation", torch.zeros((num_layers, hidden_dim))
        )

    def summarize_history(
        self, observation: Observation, action: Optional[Action]
    ) -> torch.Tensor:
        observation = torch.tensor(observation).float().view((1, self.observation_dim))
        if action is None:
            action = self.default_action
        action = torch.tensor(action).float().view((1, self.action_dim))
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
        out, (h, c) = self.lstm(
            observation_action_pair,
            (self.hidden_representation, self.cell_representation),
        )
        self.hidden_representation = h
        self.cell_representation = c
        return out.squeeze(0)

    def get_history(self) -> torch.Tensor:
        return self.history

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h0 = self.hidden_representation.unsqueeze(1).repeat(1, batch_size, 1).detach()
        c0 = self.cell_representation.unsqueeze(1).repeat(1, batch_size, 1).detach()
        out, (_, _) = self.lstm(x, (h0, c0))
        return out[:, -1, :].view((batch_size, -1))

    def reset(self) -> None:
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.action_dim + self.observation_dim)),
        )
        self.register_buffer(
            "cell_representation", torch.zeros((self.num_layers, self.hidden_dim))
        )
        self.register_buffer(
            "hidden_representation", torch.zeros((self.num_layers, self.hidden_dim))
        )
