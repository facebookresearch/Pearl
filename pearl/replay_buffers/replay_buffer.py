from abc import ABC, abstractmethod
from typing import Any

import torch


class ReplayBuffer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._is_action_continuous: bool = False
        self._has_cost_available: bool = False

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @device.setter
    @abstractmethod
    def device(self, new_device: torch.device) -> None:
        pass

    @abstractmethod
    def push(self, *args: Any) -> None:
        """Saves a transition."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> object:
        pass

    @abstractmethod
    def clear(self) -> None:
        """Empties replay buffer"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def is_action_continuous(self) -> bool:
        """Whether the action space is continuous or not."""
        return self._is_action_continuous

    @is_action_continuous.setter
    def is_action_continuous(self, value: bool) -> None:
        """Set whether the action space is continuous or not."""
        self._is_action_continuous = value

    @property
    def has_cost_available(self) -> bool:
        return self._has_cost_available

    @has_cost_available.setter
    def has_cost_available(self, value: bool) -> None:
        self._has_cost_available = value
