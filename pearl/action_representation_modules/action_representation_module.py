from abc import ABC, abstractmethod

import torch

import torch.nn as nn


class ActionRepresentationModule(ABC, nn.Module):
    """
    An abstract interface for action representation module.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
