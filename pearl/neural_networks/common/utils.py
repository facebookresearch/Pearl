# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import logging
from typing import Any
from enum import Enum

import torch
import torch.nn as nn

from pearl.neural_networks.common.residual_wrapper import ResidualWrapper

from torch.func import stack_module_state


class NormalizedSoftplus(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super(NormalizedSoftplus, self).__init__()
        self.dim: int = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = nn.functional.softplus(x)
        return values / torch.sum(values, dim=self.dim, keepdim=True)


# Activations and loss functions
class ActivationType(Enum):
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    SOFTMAX = "softmax"
    NORMALIZED_SOFTPLUS = "normalized_softplus"

    def module(self) -> nn.Module:
        if self is ActivationType.TANH:
            return nn.Tanh()
        if self is ActivationType.RELU:
            return nn.ReLU()
        if self is ActivationType.LEAKY_RELU:
            return nn.LeakyReLU()
        if self is ActivationType.LINEAR:
            return nn.Identity()
        if self is ActivationType.SIGMOID:
            return nn.Sigmoid()
        if self is ActivationType.SOFTPLUS:
            return nn.Softplus()
        if self is ActivationType.SOFTMAX:
            return nn.Softmax(dim=-1)
        if self is ActivationType.NORMALIZED_SOFTPLUS:
            return NormalizedSoftplus()
        raise ValueError(f"Unhandled activation type {self}")


class LossType(Enum):
    MSE = "mse"
    MAE = "mae"
    CROSS_ENTROPY = "cross_entropy"

    def function(self) -> Any:
        if self is LossType.MSE:
            return nn.functional.mse_loss
        if self is LossType.MAE:
            return nn.functional.l1_loss
        if self is LossType.CROSS_ENTROPY:
            return nn.functional.binary_cross_entropy
        raise ValueError(f"Unhandled loss type {self}")


def mlp_block(
    input_dim: int,
    hidden_dims: list[int] | None,
    output_dim: int = 1,
    use_batch_norm: bool = False,
    use_layer_norm: bool = False,
    hidden_activation: str = "relu",
    last_activation: str | None = None,
    dropout_ratio: float = 0.0,
    use_skip_connections: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    A simple MLP which can be reused to create more complex networks
    Args:
        input_dim: dimension of the input layer
        hidden_dims: a list of dimensions of the hidden layers
        output_dim: dimension of the output layer
        use_batch_norm: whether to use batch_norm or not in the hidden layers
        hidden_activation: activation function used for hidden layers
        last_activation: this is optional, if need activation for layer, set this input
                        otherwise, no activation is applied on last layer
        dropout_ratio: user needs to call nn.Module.eval to ensure dropout is ignored during act
    Returns:
        an nn.Sequential module consisting of mlp layers
    """
    if hidden_dims is None:
        hidden_dims = []
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(len(dims) - 2):
        single_layers = []
        input_dim_current_layer = dims[i]
        output_dim_current_layer = dims[i + 1]
        single_layers.append(
            nn.Linear(input_dim_current_layer, output_dim_current_layer)
        )
        if use_layer_norm:
            single_layers.append(nn.LayerNorm(output_dim_current_layer))
        if dropout_ratio > 0:
            single_layers.append(nn.Dropout(p=dropout_ratio))
        single_layers.append(ActivationType(hidden_activation).module())
        if use_batch_norm:
            single_layers.append(nn.BatchNorm1d(output_dim_current_layer))
        single_layer_model = nn.Sequential(*single_layers)
        if use_skip_connections:
            if input_dim_current_layer == output_dim_current_layer:
                single_layer_model = ResidualWrapper(single_layer_model)
            else:
                logging.warning(
                    "Skip connections are enabled, "
                    f"but layer in_dim ({input_dim_current_layer}) != out_dim "
                    f"({output_dim_current_layer})."
                    "Skip connection will not be added for this layer"
                )
        layers.append(single_layer_model)

    last_layer = []
    last_layer.append(nn.Linear(dims[-2], dims[-1]))
    if last_activation is not None:
        last_layer.append(ActivationType(last_activation).module())
    last_layer_model = nn.Sequential(*last_layer)
    if use_skip_connections:
        if dims[-2] == dims[-1]:
            last_layer_model = ResidualWrapper(last_layer_model)
        else:
            logging.warning(
                "Skip connections are enabled, "
                f"but layer in_dim ({dims[-2]}) != out_dim ({dims[-1]}). "
                "Skip connection will not be added for this layer"
            )
    layers.append(last_layer_model)
    return nn.Sequential(*layers)


def conv_block(
    input_channels_count: int,
    output_channels_list: list[int],
    kernel_sizes: list[int],
    strides: list[int],
    paddings: list[int],
    use_batch_norm: bool = False,
) -> nn.Module:
    """
    Reminder: torch.Conv2d layers expect inputs as (batch_size, in_channels, height, width)
    Notes: layer norm is typically not used with CNNs

    Args:
        input_channels_count: number of input channels
        output_channels_list: a list of number of output channels for each convolutional layer
        kernel_sizes: a list of kernel sizes for each layer
        strides: a list of strides for each layer
        paddings: a list of paddings for each layer
        use_batch_norm: whether to use batch_norm or not in the convolutional layers
    Returns:
        an nn.Sequential module consisting of convolutional layers
    """
    layers = []
    for out_channels, kernel_size, stride, padding in zip(
        output_channels_list, kernel_sizes, strides, paddings
    ):
        conv_layer = nn.Conv2d(
            input_channels_count,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        layers.append(conv_layer)
        if use_batch_norm:
            # batch norm should normalize the output of the convolutional layer
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        # number of input channels to next layer is number of output channels of previous layer:
        input_channels_count = out_channels

    return nn.Sequential(*layers)


def xavier_init_weights(m: nn.Module) -> None:
    """Initialize Linear layer weights with Xavier uniform initializer."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def uniform_init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.001, 0.001)
        nn.init.uniform_(m.bias, -0.001, 0.001)


def update_target_network(
    target_network: nn.Module, source_network: nn.Module, tau: float
) -> None:
    # Q_target = (1 - tao) * Q_target + tao*Q
    for target_param, source_param in zip(
        target_network.parameters(), source_network.parameters()
    ):
        if target_param is source_param:
            # skip soft-updating when the target network shares the parameter with
            # the network being train.
            continue
        new_param = tau * source_param.data + (1.0 - tau) * target_param.data
        target_param.data.copy_(new_param)


def ensemble_forward(
    models: nn.ModuleList | list[nn.Module],
    features: torch.Tensor,
    use_for_loop: bool = True,
) -> torch.Tensor:
    """
    Run forward pass on several models and return their outputs stacked as a tensor.
    If use_for_loop is False, a vectorized implementation is used, which has some
        limitations (see https://fburl.com/code/m4l2tjof):
    1. All models must have the same structure.
    2. Gradient backpropagation to original model parameters might not work properly.

    Args:
        models: list of models to run forward pass on. Length: num_models
        features: features to run forward pass on. shape: (batch_size, num_models, num_features)
        use_for_loop: whether to use for loop or vectorized implementation
    Output:
        A tensor of shape (batch_size, num_models)
    """
    torch._assert(
        features.ndim == 3,
        "Features should be of shape (batch_size, num_models, num_features)",
    )
    torch._assert(
        features.shape[1] == len(models),
        "Number of models must match features.shape[1]",
    )
    batch_size = features.shape[0]

    if use_for_loop:
        values = [model(features[:, i, :]).flatten() for i, model in enumerate(models)]
        return torch.stack(values, dim=-1)  # (batch_size, ensemble_size)
    else:
        features = features.permute((1, 0, 2))

        def wrapper(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            data: torch.Tensor,
        ) -> torch.Tensor:
            return torch.func.functional_call(models[0], (params, buffers), data)

        params, buffers = stack_module_state(models)
        values = torch.vmap(wrapper)(params, buffers, features).view(
            (-1, batch_size)
        )  # (ensemble_size, batch_size)

        # change shape to (batch_size, ensemble_size)
        return values.permute(1, 0)


def update_target_networks(
    list_of_target_networks: nn.ModuleList | list[nn.Module],
    list_of_source_networks: nn.ModuleList | list[nn.Module],
    tau: float,
) -> None:
    """
    Args:
        list_of_target_networks: nn.ModuleList() of nn.Module()
        list_of_source_networks: nn.ModuleList() of nn.Module()
        tau: parameter for soft update
    """
    # Q_target = (1 - tao) * Q_target + tao*Q
    for target_network, source_network in zip(
        list_of_target_networks, list_of_source_networks
    ):
        update_target_network(target_network, source_network, tau)


def compute_output_dim_model_cnn(
    input_channels: int,
    input_width: int,
    input_height: int,
    model_cnn: nn.Module,
) -> int:
    """Return flattened output dimension of a CNN block.

    Parameters correspond to the input tensor layout
    ``(batch_size, input_channels, input_height, input_width)``.
    """
    dummy_input = torch.zeros(1, input_channels, input_height, input_width)
    with torch.no_grad():
        dummy_output_flattened = torch.flatten(
            model_cnn(dummy_input), start_dim=1, end_dim=-1
        )
    return dummy_output_flattened.shape[1]
