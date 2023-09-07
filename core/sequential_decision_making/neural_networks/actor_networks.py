"""
This module defines several types of actor neural networks.

Constants:
    ActorNetworkType: a type (and therefore a callable) getting state_dim, hidden_dims, output_dim and producing a neural network with
    able to produce an action probability given a state.
"""


from typing import Callable, List

import torch

import torch.nn as nn
import torch.nn.functional as F
from pearl.core.common.neural_networks.auto_device_nn_module import AutoDeviceNNModule
from pearl.core.common.neural_networks.utils import mlp_block


class VanillaActorNetwork(AutoDeviceNNModule):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(VanillaActorNetwork, self).__init__()
        self._model = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            last_activation="softmax",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class VanillaContinuousActorNetwork(AutoDeviceNNModule):
    """
    This is vanilla version of deterministic actor network
    Given input state, output an action vector
    Args
        output_dim: action dimension
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        super(VanillaContinuousActorNetwork, self).__init__()
        self._model = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            last_activation="tanh",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


ActorNetworkType = Callable[[int, int, List[int], int], nn.Module]
