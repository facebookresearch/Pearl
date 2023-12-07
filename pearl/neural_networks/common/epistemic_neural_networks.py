# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This module defines epistemic neural networks that can model posterior distributions
    and perform bayesian updates via deep learning
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn
from pearl.neural_networks.common.utils import mlp_block
from torch import Tensor


class EpistemicNeuralNetwork(ABC, nn.Module):
    """
    Epistemic Neural Network is an abstract neural network model that can model posterior of
        the expectation of a random variable, i.e. epistemic uncertainty.
    Args:
        input_dim: int. Input feature dimension.
        hidden_dims: List[int]. Hidden layer dimensions.
        output_dim: int. Output dimension.
    """

    def __init__(
        self, input_dim: int, hidden_dims: Optional[List[int]], output_dim: int = 1
    ) -> None:
        super(EpistemicNeuralNetwork, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """
        Input:
            x: Feature vector of state action pairs
            z: Optional. Epistemic indices
        Output:
            posterior samples (corresponding to index z if available)
        """
        pass


class MLPWithPrior(nn.Module):
    """
    MLP model with prior regularization.
    If used as a single model, this is equivalent to L2 regularization.
    Args:
        input_dim (int): The dimension of the input features.
        hidden_dims (List[int]): A list containing the dimensions of the hidden layers.
        output_dim (int): The dimension of the output.
        scale (float): The scale of the prior regularization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int = 1,
        scale: float = 1.0,
    ) -> None:
        super(MLPWithPrior, self).__init__()
        self.base_net: nn.Module = mlp_block(input_dim, hidden_dims, output_dim)
        self.prior_net: nn.Module = mlp_block(input_dim, hidden_dims, output_dim).eval()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:
            x: Tensor. Feature vector input
        Output:
            regularized sample output from posterior
        """
        with torch.no_grad():
            prior = self.scale * self.prior_net(x)

        return self.base_net(x) + prior


class Ensemble(EpistemicNeuralNetwork):
    """
    An ensemble based implementation of epistemic neural network.
    Args:
        input_dim: int. Input feature dimension.
        hidden_dims: List[int]. Hidden layer dimensions.
        output_dim: int. Output dimension.
        ensemble_size: int. Number of particles in the ensemble
                            to construct posterior.
        prior_scale: float. prior regularization scale.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int = 1,
        ensemble_size: int = 10,
        prior_scale: float = 1.0,
    ) -> None:
        super(Ensemble, self).__init__(input_dim, hidden_dims, output_dim)
        self.ensemble_size = ensemble_size

        self.models = nn.ModuleList(
            [
                MLPWithPrior(input_dim, hidden_dims, output_dim, scale=prior_scale)
                for _ in range(ensemble_size)
            ]
        )

        self._resample_epistemic_index()

    def forward(
        self, x: Tensor, z: Optional[Tensor] = None, persistent: bool = False
    ) -> Tensor:
        """
        Input:
            x: Feature vector of state action pairs
            z: Single integer tensor. Ensemble epistemic index
        Output:
            posterior samples corresponding to z
        """
        if z is not None:
            assert z.flatten().shape[0] == 1
            ensemble_index = int(z.item())
            assert ensemble_index >= 0 and ensemble_index < self.ensemble_size
        else:
            ensemble_index = self.z

        return self.models[ensemble_index](x)

        if not persistent:
            self._resample_epistemic_index()

    def _resample_epistemic_index(self) -> None:
        self.z = torch.randint(0, self.ensemble_size, (1,))
