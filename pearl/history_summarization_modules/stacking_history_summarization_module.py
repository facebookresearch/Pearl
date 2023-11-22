from typing import Optional

import torch
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)


class StackingHistorySummarizationModule(HistorySummarizationModule):
    """
    A history summarization module that simply stacks observations into a history.
    """

    def __init__(
        self, observation_dim: int, action_dim: int, history_length: int = 8
    ) -> None:
        super(StackingHistorySummarizationModule, self).__init__()
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.register_buffer("default_action", torch.zeros((1, action_dim)))
        self.register_buffer(
            "history", torch.zeros((history_length, action_dim + observation_dim))
        )

    def summarize_history(
        self, observation: torch.Tensor, action: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if action is None:
            action = self.default_action

        assert observation.shape[-1] + action.shape[-1] == self.history.shape[-1]
        observation_action_pair = torch.cat((action, observation), dim=-1).detach()
        self.history = torch.cat(
            [
                self.history[1:, :],
                observation_action_pair.view(
                    (1, self.action_dim + self.observation_dim)
                ),
            ],
            dim=0,
        )
        return self.history.view((-1))

    def get_history(self) -> torch.Tensor:
        return self.history.view((-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reset(self) -> None:
        self.register_buffer(
            "history",
            torch.zeros((self.history_length, self.action_dim + self.observation_dim)),
        )
