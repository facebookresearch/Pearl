# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

"""
This module defines epistemic neural networks that can model posterior distributions
    and perform bayesian updates via deep learning
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from pearl.neural_networks.common.utils import init_weights, mlp_block
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
        self, input_dim: int, hidden_dims: list[int] | None, output_dim: int = 1
    ) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor, z: Tensor) -> Tensor:
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
        hidden_dims: list[int] | None,
        output_dim: int = 1,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
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
        hidden_dims: list[int] | None,
        output_dim: int = 1,
        ensemble_size: int = 10,
        prior_scale: float = 1.0,
    ) -> None:
        super().__init__(input_dim, hidden_dims, output_dim)
        self.ensemble_size = ensemble_size

        self.models = nn.ModuleList(
            [
                MLPWithPrior(input_dim, hidden_dims, output_dim, scale=prior_scale)
                for _ in range(ensemble_size)
            ]
        )

        self._resample_epistemic_index()

    def forward(self, x: Tensor, z: Tensor, persistent: bool = False) -> Tensor:
        """
        Input:
            x: Feature vector of state action pairs
            z: Single integer tensor. Ensemble epistemic index
        Output:
            posterior samples corresponding to z
        """
        assert z.flatten().shape[0] == 1
        ensemble_index = int(z.item())
        assert ensemble_index >= 0 and ensemble_index < self.ensemble_size

        if not persistent:
            self._resample_epistemic_index()

        return self.models[ensemble_index](x)

    def _resample_epistemic_index(self) -> None:
        self.z = torch.randint(0, self.ensemble_size, (1,))


class Priornet(nn.Module):
    """
    Prior network for epinet.  This network contains an ensemble of
    randomly initialized models which are held fixed during training.
    """

    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int, index_dim: int
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.index_dim = index_dim
        models = []
        for _ in range(self.index_dim):
            model = mlp_block(self.input_dim, self.hidden_dims, self.output_dim)
            # Xavier uniform initalization
            model.apply(init_weights)
            models.append(model)
        self.base_model: nn.Module = mlp_block(
            self.input_dim, self.hidden_dims, self.output_dim
        )
        self.base_model.apply(init_weights)
        self.base_model = self.base_model.to("meta")
        self.models: nn.ModuleList = nn.ModuleList(models)

        self.params: dict[str, Any]
        self.buffers: dict[str, Any]
        self.generate_params_buffers()

    def generate_params_buffers(self) -> None:
        """
        Generate parameters and buffers for the priornet.
        """
        self.params, self.buffers = torch.func.stack_module_state(self.models)

    def call_single_model(
        self, params: dict[str, Any], buffers: dict[str, Any], data: Tensor
    ) -> Tensor:
        """
        Method for parallelizing priornet forward passes with torch.vmap.
        """
        return torch.func.functional_call(self.base_model, (params, buffers), (data,))

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Perform forward pass on the priornet ensemble and weight by epistemic index
        x and z are assumed to already be formatted.
        Input:
            x: tensor consisting of concatentated input and epistemic index
            z: tensor consisting of epistemic index
        Output:
            ensemble output of x weighted by epistemic index vector z.
        """
        # vmap is not compatible with torchscript
        # outputs = torch.vmap(self.call_single_model, (0, 0, None))(
        #     self.params, self.buffers, x
        # )
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        return torch.einsum("ijk,ji->jk", outputs, z)


class Epinet(EpistemicNeuralNetwork):
    def __init__(
        self,
        index_dim: int,
        input_dim: int,
        output_dim: int,
        num_indices: int,
        epi_hiddens: list[int],
        prior_hiddens: list[int],
        prior_scale: float,
    ) -> None:
        super().__init__(input_dim, None, output_dim)
        self.index_dim = index_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_indices = num_indices
        self.epi_hiddens = epi_hiddens
        self.prior_hiddens = prior_hiddens
        self.prior_scale = prior_scale

        epinet_input_dim = self.input_dim + self.index_dim
        # Trainable Epinet
        self.epinet: nn.Module = mlp_block(
            epinet_input_dim, self.epi_hiddens, self.index_dim * self.output_dim
        )
        self.epinet.apply(init_weights)
        # Priornet
        self.priornet = Priornet(
            self.input_dim, self.prior_hiddens, self.output_dim, self.index_dim
        )

    def format_xz(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Take cartesian product of x and z and concatenate for forward pass.
        Input:
            x: Feature vectors containing item and user embeddings and interactions
            z: Epinet epistemic indices
        Output:
            xz: Concatenated cartesian product of x and z
        """
        batch_size, d = x.shape
        num_indices, _ = z.shape
        x_expanded = x.unsqueeze(1).expand(batch_size, num_indices, d)
        z_expanded = z.unsqueeze(0).expand(batch_size, num_indices, self.index_dim)
        xz = torch.cat([x_expanded, z_expanded], dim=-1)
        return xz.view(batch_size * num_indices, d + self.index_dim)

    def forward(self, x: Tensor, z: Tensor, persistent: bool = False) -> Tensor:
        """
        Input:
            x: Feature vector containing item and user embeddings and interactions
            z: Matrix containing . Epinet epistemic indices
        Output:
            posterior samples corresponding to z
        """
        xz = self.format_xz(x, z)
        x_cartesian, z_cartesian = xz[:, : -self.index_dim], xz[:, -self.index_dim :]
        batch_size, _ = xz.shape
        epinet_out = self.epinet(xz.detach()).view(
            batch_size, self.output_dim, self.index_dim
        )
        epinet_out = torch.einsum("ijk,ik->ij", epinet_out, z_cartesian)
        with torch.no_grad():
            priornet_out = self.prior_scale * self.priornet(x_cartesian, z_cartesian)
        return epinet_out + priornet_out
