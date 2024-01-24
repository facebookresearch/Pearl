# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.neural_networks.contextual_bandit.base_cb_model import MuSigmaCBModel
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.policy_learners.contextual_bandits.neural_bandit import ACTIVATION_MAP
from torch.nn.modules.activation import LeakyReLU, ReLU, Sigmoid, Softplus, Tanh


logger: logging.Logger = logging.getLogger(__name__)


class NeuralLinearRegression(MuSigmaCBModel):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int],  # last one is the input dim for linear regression
        l2_reg_lambda_linear: float = 1.0,
        output_activation_name: str = "linear",
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        hidden_activation: str = "relu",
        last_activation: Optional[str] = None,
        dropout_ratio: float = 0.0,
        use_skip_connections: bool = True,
    ) -> None:
        """
        A model for Neural LinUCB (can also be used for Neural LinTS).
        It's a NN with a linear regression layer at the end.
        Based on the paper https://arxiv.org/pdf/2012.01780.pdf

        Args:
            feature_dim: number of features
            hidden_dims: size of hidden layers in the network
            l2_reg_lambda_linear: L2 regularization parameter for the linear regression layer
            output_activation_name: output activation function name (see ACTIVATION_MAP)
            use_batch_norm: whether to use batch normalization
            use_layer_norm: whether to use layer normalization
            hidden_activation: activation function for hidden layers
            last_activation: activation function for the last layer
            dropout_ratio: dropout ratio
            use_skip_connections: whether to use skip connections
        """
        super(NeuralLinearRegression, self).__init__(feature_dim=feature_dim)
        self._nn_layers = VanillaValueNetwork(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            hidden_activation=hidden_activation,
            last_activation=last_activation,
            dropout_ratio=dropout_ratio,
            use_skip_connections=use_skip_connections,
        )
        self._linear_regression_layer = LinearRegression(
            feature_dim=hidden_dims[-1],
            l2_reg_lambda=l2_reg_lambda_linear,
        )
        self.output_activation: Union[
            LeakyReLU, ReLU, Sigmoid, Softplus, Tanh, nn.Identity
        ] = ACTIVATION_MAP[output_activation_name]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be [batch_size, feature_dim] or [batch_size, num_arms, feature_dim]
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]

        # dim: [batch_size * num_arms, feature_dim]
        x = x.reshape(-1, feature_dim)

        # dim: [batch_size * num_arms, 1]
        x = self._nn_layers(x)  # apply NN layers
        x = self._linear_regression_layer(x)  # apply linear regression to NN output
        x = self.output_activation(x)  # apply output activation

        # dim: [batch_size, num_arms]
        return x.reshape(batch_size, -1)

    def calculate_sigma(self, x: torch.Tensor) -> torch.Tensor:
        # x can be [batch_size, feature_dim] or [batch_size, num_arms, feature_dim]
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]

        # dim: [batch_size * num_arms, feature_dim]
        x = x.reshape(-1, feature_dim)
        # dim: [batch_size * num_arms, 1]
        x = self._nn_layers(x)

        return self._linear_regression_layer.calculate_sigma(x).reshape(batch_size, -1)

    def forward_with_intermediate_values(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass, which returns not only the final prediction, but also the intermediate
            values (output of NN layers).
        """
        # x can be [batch_size, feature_dim] or [batch_size, num_arms, feature_dim]
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]

        # dim: [batch_size * num_arms, feature_dim]
        x = x.reshape(-1, feature_dim)

        # dim: [batch_size * num_arms, NN_out_dim]
        nn_output = self._nn_layers(x)

        # dim: [batch_size * num_arms, 1]
        x = self._linear_regression_layer(nn_output)

        # dim: [batch_size, num_arms]
        return {
            "pred_label": self.output_activation(x).reshape(batch_size, -1),
            "nn_output": nn_output,
        }
