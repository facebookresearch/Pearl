# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

"""
This module defines several types of value neural networks.
"""

from abc import ABC
from typing import Any

import torch
import torch.nn as nn

from pearl.neural_networks.common.utils import (
    compute_output_dim_model_cnn,
    conv_block,
    mlp_block,
)
from torch import Tensor


class ValueNetwork(nn.Module, ABC):
    """
    An interface for value neural networks.
    It does not add any required methods to those already present in
    its super classes.
    Its purpose instead is just to serve as an umbrella type for all value networks.
    """


class VanillaValueNetwork(ValueNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None,
        output_dim: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    # default initialization in linear and conv layers of a F.sequential model is Kaiming
    def xavier_init(self) -> None:
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        for layer in self._model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)


class CNNValueNetwork(ValueNetwork):
    """
    Vanilla CNN with a convolutional block followed by an mlp block.
    Args:
        input_width: width of the input
        input_height: height of the input
        input_channels_count: number of input channels
        kernel_sizes: list of kernel sizes for the convolutional layers
        output_channels_list: list of number of output channels for each convolutional layer
        strides: list of strides for each layer
        paddings: list of paddings for each layer
        hidden_dims_fully_connected: a list of dimensions of the hidden layers in the mlp
        use_batch_norm_conv: whether to use batch_norm in the convolutional layers
        use_batch_norm_fully_connected: whether to use batch_norm in the fully connected layers
        output_dim: dimension of the output layer
    Returns:
        An nn.Sequential module consisting of a convolutional block followed by an mlp.
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_channels_count: int,
        kernel_sizes: list[int],
        output_channels_list: list[int],
        strides: list[int],
        paddings: list[int],
        hidden_dims_fully_connected: None
        | (list[int]) = None,  # hidden dims for fully connected layers
        use_batch_norm_conv: bool = False,
        use_batch_norm_fully_connected: bool = False,
        output_dim: int = 1,  # dimension of the final output
    ) -> None:
        assert (
            len(kernel_sizes)
            == len(output_channels_list)
            == len(strides)
            == len(paddings)
        )
        super(CNNValueNetwork, self).__init__()

        self._input_channels = input_channels_count
        self._input_height = input_height
        self._input_width = input_width
        self._output_channels = output_channels_list
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._paddings = paddings
        if hidden_dims_fully_connected is None:
            self._hidden_dims_fully_connected: list[int] = []
        else:
            self._hidden_dims_fully_connected: list[int] = hidden_dims_fully_connected

        self._use_batch_norm_conv = use_batch_norm_conv
        self._use_batch_norm_fully_connected = use_batch_norm_fully_connected
        self._output_dim = output_dim

        self._model_cnn: nn.Module = conv_block(
            input_channels_count=self._input_channels,
            output_channels_list=self._output_channels,
            kernel_sizes=self._kernel_sizes,
            strides=self._strides,
            paddings=self._paddings,
            use_batch_norm=self._use_batch_norm_conv,
        )

        self._mlp_input_dims: int = compute_output_dim_model_cnn(
            input_channels=input_channels_count,
            input_width=input_width,
            input_height=input_height,
            model_cnn=self._model_cnn,
        )
        self._model_fc: nn.Module = mlp_block(
            input_dim=self._mlp_input_dims,
            hidden_dims=self._hidden_dims_fully_connected,
            output_dim=self._output_dim,
            use_batch_norm=self._use_batch_norm_fully_connected,
        )

    def forward(self, x: Tensor) -> Tensor:
        out_cnn = self._model_cnn(x / 255.0)
        out_flattened = torch.flatten(out_cnn, start_dim=1, end_dim=-1)
        out_fc = self._model_fc(out_flattened)
        return out_fc
