import torch
from pearl.api.observation import Observation
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)


class StackingHistorySummarizationModule(HistorySummarizationModule):
    """
    A history summarization module that simply stacks observations into a history.
    """

    def __init__(self, observation_dim: int, history_length: int = 8) -> None:
        super(StackingHistorySummarizationModule, self).__init__()
        self.history_length = history_length
        self.observation_dim = observation_dim
        self.history: torch.Tensor = torch.zeros((history_length, observation_dim))

    def summarize_history(self, observation: Observation) -> torch.Tensor:
        observation = torch.tensor(observation).float()
        assert observation.shape[-1] == self.history.shape[-1]
        self.history = torch.cat(
            [
                observation.view((1, self.observation_dim)),
                self.history[1:, :],
            ],
            dim=0,
        )
        return self.history.view((-1))

    def get_history(self) -> torch.Tensor:
        return self.history.view((-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
