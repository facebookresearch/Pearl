# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import logging
from typing import List

import torch
import torch.nn as nn
from pearl.neural_networks.common.utils import ACTIVATION_MAP
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.neural_networks.contextual_bandit.base_cb_model import MuSigmaCBModel
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.utils.module_utils import modules_have_similar_state_dict

logger: logging.Logger = logging.getLogger(__name__)


class NeuralLinearRegression(MuSigmaCBModel):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: list[int],  # last one is the input dim for linear regression
        l2_reg_lambda_linear: float = 1.0,
        gamma: float = 1.0,
        force_pinv: bool = False,
        output_activation_name: str = "linear",
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        hidden_activation: str = "relu",
        last_activation: str | None = None,
        dropout_ratio: float = 0.0,
        use_skip_connections: bool = True,
        nn_e2e: bool = True,
    ) -> None:
        """
        A model for Neural LinUCB (can also be used for Neural LinTS).
        It's a NN with a linear regression layer at the end.
        Based on the paper https://arxiv.org/pdf/2012.01780.pdf

        Args:
            feature_dim: number of features
            hidden_dims: size of hidden layers in the network
            l2_reg_lambda_linear: L2 regularization parameter for the linear regression layer
            gamma: discounting multiplier for the linear regression layer
            force_pinv: If True, we will always use pseudo inverse to invert the `A` matrix. If
                False, we will first try to use regular matrix inversion. If it fails, we will
                fallback to pseudo inverse.
            output_activation_name: output activation function name (see ACTIVATION_MAP)
            use_batch_norm: whether to use batch normalization
            use_layer_norm: whether to use layer normalization
            hidden_activation: activation function for hidden layers
            last_activation: activation function for the last layer
            dropout_ratio: dropout ratio
            use_skip_connections: whether to use skip connections
            nn_e2e: If True, we use a Linear NN layer to generate mu instead of getting it from
                LinUCB. This can improve learning stability. Sigma is still generated from LinUCB.

        """
        super().__init__(feature_dim=feature_dim)
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
            gamma=gamma,
            force_pinv=force_pinv,
        )
        self.output_activation: nn.Module = ACTIVATION_MAP[output_activation_name]()
        self.linear_layer_e2e = nn.Linear(
            in_features=hidden_dims[-1], out_features=1, bias=False
        )  # used only if nn_e2e is True
        self.nn_e2e = nn_e2e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be [batch_size, feature_dim] or [batch_size, num_arms, feature_dim]
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]

        # dim: [batch_size * num_arms, feature_dim]
        x = x.reshape(-1, feature_dim)

        # dim: [batch_size * num_arms, 1]
        x = self._nn_layers(x)  # apply NN layers

        if self.nn_e2e:
            # get mu from end-to-end NN
            x = self.linear_layer_e2e(x)  # apply linear layer to NN output
        else:
            # get mu from LinUCB
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
    ) -> dict[str, torch.Tensor]:
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
        if self.nn_e2e:
            # get mu from end-to-end NN
            x = self.linear_layer_e2e(nn_output)  # apply linear layer to NN output
        else:
            # get mu from LinUCB
            x = self._linear_regression_layer(
                nn_output
            )  # apply linear regression to NN output

        # dim: [batch_size, num_arms]
        return {
            "pred_label_pre_activation": x.reshape(batch_size, -1),
            "pred_label": self.output_activation(x).reshape(batch_size, -1),
            "nn_output": nn_output,
        }

    def compare(self, other: MuSigmaCBModel) -> str:
        """
        Compares to a LinearRegression instance for equality,
        checking attributes, NN layers, and the linear regression layer.

        Args:
        other: The other NeuralLinearRegression instance to compare with.

        Returns:
        str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, NeuralLinearRegression):
            differences.append("other is not an instance of NeuralLinearRegression")
        assert isinstance(other, NeuralLinearRegression)

        # Compare attributes
        if self.nn_e2e != other.nn_e2e:
            differences.append(f"nn_e2e is different: {self.nn_e2e} vs {other.nn_e2e}")

        if type(self.output_activation) is not type(other.output_activation):
            differences.append(
                f"output_activation is different: {self.output_activation} "
                + f"vs {other.output_activation}"
            )

        # Compare NN layers using modules_have_similar_state_dict
        if (
            reason := modules_have_similar_state_dict(self._nn_layers, other._nn_layers)
        ) != "":
            differences.append(f"_nn_layers are different: {reason}")

        # Compare linear regression layers using their compare method
        if (
            reason := self._linear_regression_layer.compare(
                other._linear_regression_layer
            )
        ) != "":
            differences.append(f"_linear_regression_layer is different: {reason}")

        # Compare linear layer for e2e case
        if (
            reason := modules_have_similar_state_dict(
                self.linear_layer_e2e, other.linear_layer_e2e
            )
        ) != "":
            differences.append(f"linear_layer_e2e is different: {reason}")

        return "\n".join(differences)  # Join the differences with newlines
