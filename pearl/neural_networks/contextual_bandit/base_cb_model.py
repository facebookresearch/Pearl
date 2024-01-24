from abc import abstractmethod

import torch
import torch.nn as nn


class MuSigmaCBModel(nn.Module):
    """
    A base class for CB models that output both mu (point prediction) and sigma (standard deviation
        of prediction uncertainty).

    Args:
        feature_dim: number of features

    Methods:
        forward: forward pass. Returns a expected reward
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self._feature_dim = feature_dim

    # TODO: do we want to force the input to be a single tensor?
    # We can change this later if we need a different input structure.
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
