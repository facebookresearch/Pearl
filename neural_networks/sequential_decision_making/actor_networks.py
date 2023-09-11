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
from pearl.neural_networks.common.auto_device_nn_module import AutoDeviceNNModule
from pearl.neural_networks.common.utils import mlp_block
from torch.distributions import Normal


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


class ContinousActorNetwork(AutoDeviceNNModule):
    def __init__(self, input_dim, hidden_dims, output_dim, action_bound):
        super(ContinousActorNetwork, self).__init__()
        self.action_bound = action_bound
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        hiddens = []
        for i in range(len(hidden_dims) - 1):
            hiddens.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hiddens.append(nn.ReLU())
        self.hiddens = nn.Sequential(*hiddens)

        self.fc_mu = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.fc_std = torch.nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # TODO : could use some improvements such as outputs deterministic etc, example refers to https://github.com/jparkerholder/SAC-PyTorch/blob/master/sac.py
        x = F.relu(self.fc1(x))  # noqa: F821
        x = self.hiddens(x)

        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))  # noqa: F821
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample() conducts reparameterization
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        action = action * self.action_bound
        return action, log_prob


ActorNetworkType = Callable[[int, int, List[int], int], nn.Module]
