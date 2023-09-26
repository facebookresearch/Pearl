"""
This module defines several types of actor neural networks.

Constants:
    ActorNetworkType: a type (and therefore a callable) getting state_dim, hidden_dims, output_dim and producing a neural network with
    able to produce an action probability given a state.
"""


from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.auto_device_nn_module import AutoDeviceNNModule
from pearl.neural_networks.common.utils import mlp_block

from pearl.utils.device import get_pearl_device
from torch import Tensor
from torch.distributions import Normal


def scale_action(
    action_space: ActionSpace, normalized_action: torch.Tensor
) -> torch.Tensor:
    """
    For continous action spaces, actor networks output "normalized_actions", i.e. actions in the range [-1, 1]^{action_dim}
    This function rescales the normalized_action from [-1, 1]^{action_dim} to [low, high]^{action_dim}
    - Note: while the action space is not assumed to be symmetric (low = -high), it is assumed that
            low and high are the same for each dimension (as is the case for most gym environments).
    """
    device = get_pearl_device()
    low, high = torch.tensor(action_space.low).to(device), torch.tensor(
        action_space.high
    ).to(device)
    return low + (0.5 * (normalized_action + 1.0) * (high - low))


class VanillaActorNetwork(AutoDeviceNNModule):
    def __init__(self, input_dim, hidden_dims, output_dim, action_space=None):
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

    def __init__(self, input_dim, hidden_dims, output_dim, action_space=None):
        super(VanillaContinuousActorNetwork, self).__init__()
        self._model = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            last_activation="tanh",
        )

    # the forward function should probably be renamed to sample_action
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class GaussianActorNetwork(AutoDeviceNNModule):
    """
    A multivariate gaussian actor network: parameterize the policy (action distirbution) as a multivariate gaussian.
    Given input state, the network outputs a pair of mu and sigma, where mu is the mean of the Gaussian distribution,
    and sigma is its standard deviation along different dimension
       - Note: the action distribution is assumed to be independent across different action dimensions
    Args:
        input_dim: input state dimension
        hidden_dims: list of hidden layer dimensions; cannot pass an empty list
        output_dim: action dimension
        action_space: action space
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        action_space: ActionSpace,
    ):
        super(GaussianActorNetwork, self).__init__()
        if len(hidden_dims) < 1:
            raise ValueError(
                "The hidden dims cannot be empty for a gaussian actor network."
            )

        self._model = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
        )

        self.fc_mu = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.fc_std = torch.nn.Linear(hidden_dims[-1], output_dim)
        self._action_space = action_space
        # check this for multi-dimensional spaces
        self._action_bound = (action_space.high[0] - action_space.low[0]) / 2

        # preventing the actor network from learning a high entropy distribution (soft-actor critic has a
        # maximum entropy regularization which encourgaes learning a high entropy distribution)
        self._log_std_min = -2
        self._log_std_max = 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._model(x.float())
        mean = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, min=self._log_std_min, max=self._log_std_max)
        return mean, log_std

    def sample_action_and_get_log_prob(self, state_batch: Tensor):
        epsilon = 1e-6
        mean, log_std = self.forward(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)
        sample = normal.rsample()  # reparameterization trick

        # ensure sampled action is within [-1, 1]^{action_dim}
        normalized_action = torch.tanh(sample)

        # clamp each action dimension to prevent numerical issues in tanh
        normalized_action.clamp(-1 + epsilon, 1 - epsilon)
        action = scale_action(self._action_space, normalized_action)

        log_prob = normal.log_prob(sample)
        log_prob -= torch.log(
            self._action_bound * (1 - normalized_action.pow(2)) + epsilon
        )

        # when working with multi-dimensional action space, we sum log probabilities over individual action dimension
        if log_prob.dim() == 2:
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


ActorNetworkType = Callable[[int, int, List[int], int], nn.Module]
