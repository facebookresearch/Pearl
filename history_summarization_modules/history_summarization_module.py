from abc import ABC, abstractmethod

import torch.nn as nn

from pearl.api.history import History

from pearl.api.observation import Observation

from pearl.api.state import SubjectiveState


class HistorySummarizationModule(ABC, nn.Module):
    """
    An abstract interface for exploration module.
    """

    @abstractmethod
    def summarize_history(self, observation: Observation) -> SubjectiveState:
        pass

    @abstractmethod
    def get_history(self) -> History:
        pass

    @abstractmethod
    def forward(self, x: History) -> SubjectiveState:
        pass
