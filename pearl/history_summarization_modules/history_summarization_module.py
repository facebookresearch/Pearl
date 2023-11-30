from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from pearl.api.action import Action
from pearl.api.history import History
from pearl.api.observation import Observation
from pearl.api.state import SubjectiveState


class HistorySummarizationModule(ABC, nn.Module):
    """
    An abstract interface for exploration module.
    """

    @abstractmethod
    def summarize_history(
        self, observation: Observation, action: Optional[Action]
    ) -> SubjectiveState:
        pass

    @abstractmethod
    def get_history(self) -> History:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
