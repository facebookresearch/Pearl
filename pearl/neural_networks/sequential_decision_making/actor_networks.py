"""
This module defines several types of actor neural networks.
"""


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.utils import mlp_block
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace

from torch import Tensor
from torch.distributions import Normal


def action_scaling(
    action_space: ActionSpace, input_action: torch.Tensor
) -> torch.Tensor:
    """
    Center and scale input action from [-1, 1]^{action_dim} to [low, high]^{action_dim}.
    Use cases:
        - For continuous action spaces, actor networks output "normalized_actions",
            i.e. actions in the range [-1, 1]^{action_dim}.

    Note: the action space is not assumed to be symmetric (low = -high).

    Args:
        action_space: the action space
        input_action: the input action vector to be scaled
    Returns:
        scaled_action: centered and scaled input action vector, according to the action space
    """
    assert isinstance(action_space, BoxActionSpace)
    device = input_action.device
    low = action_space.low.clone().detach().to(device)
    high = action_space.high.clone().detach().to(device)
    centered_and_scaled_action = (((high - low) * (input_action + 1.0)) / 2) + low
    return centered_and_scaled_action


def noise_scaling(action_space: ActionSpace, input_noise: torch.Tensor) -> torch.Tensor:
    """
    This function rescales any input vector from [-1, 1]^{action_dim} to [low, high]^{action_dim}.
    Use case:
        - For noise based exploration, we need to scale the noise (for example, from the standard
            normal distribution) according to the action space.

    Args:
        action_space: the action space
        input_vector: the input vector to be scaled
    Returns:
        torch.Tensor: scaled input vector, according to the action space
    """
    assert isinstance(action_space, BoxActionSpace)
    device = input_noise.device
    low = torch.tensor(action_space.low).to(device)
    high = torch.tensor(action_space.high).to(device)
    scaled_noise = ((high - low) / 2) * input_noise
    return scaled_noise


class ActorNetwork(nn.Module):
    """
    An interface for actor networks.
    IMPORTANT: the __init__ method specifies parameters for type-checking purposes only.
    It does NOT store them in attributes.
    Dealing with these parameters is left to subclasses.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int,
        action_space: Optional[ActionSpace] = None,
    ) -> None:
        super(ActorNetwork, self).__init__()


class VanillaActorNetwork(ActorNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int,
        action_space: Optional[ActionSpace] = None,
    ) -> None:
        """A Vanilla Actor Network is meant to be used with discrete action spaces.
           For an input state (batch of states), it outputs a probability distribution over
           all the actions.

        Args:
            input_dim: input state dimension (or dim of the state representation)
            hidden_dims: list of hidden layer dimensions
            output_dim: number of actions (action_space.n when used with the DiscreteActionSpace
                        class)
        """
        super(VanillaActorNetwork, self).__init__(
            input_dim, hidden_dims, output_dim, action_space
        )
        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            last_activation="softmax",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def get_action_prob(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gets probabilities of different actions from a discrete actor network.
        Assumes that the input batch of actions is one-hot encoded
            (generalize it later).

        Args:
            state_batch: batch of states with shape (batch_size, input_dim)
            action_batch: batch of actions with shape (batch_size, output_dim)
        Returns:
            action_probs: probabilities of each action in the batch with shape (batch_size)
        """
        all_action_probs = self.forward(state_batch)  # shape: (batch_size, output_dim)
        action_probs = torch.sum(all_action_probs * action_batch, dim=1, keepdim=True)

        return action_probs.view(-1)


class VanillaContinuousActorNetwork(ActorNetwork):
    """
    This is vanilla version of deterministic actor network
    Given input state, output an action vector
    Args
        output_dim: action dimension
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int,
        action_space: ActionSpace,
    ) -> None:
        super(VanillaContinuousActorNetwork, self).__init__(
            input_dim, hidden_dims, output_dim, action_space
        )
        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            last_activation="tanh",
        )
        self._action_space = action_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def sample_action(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample an action from the actor network.
        Args:
            x: input state
        Returns:
            action: sampled action, scaled to the action space bounds
        """
        normalized_action = self._model(x)
        action = action_scaling(self._action_space, normalized_action)
        return action


class GaussianActorNetwork(ActorNetwork):
    """
    A multivariate gaussian actor network: parameterize the policy (action distirbution)
    as a multivariate gaussian. Given input state, the network outputs a pair of
    (mu, sigma), where mu is the mean of the Gaussian distribution, and sigma is its
    standard deviation along different dimensions.
       - Note: action distribution is assumed to be independent across different
         dimensions
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
    ) -> None:
        super(GaussianActorNetwork, self).__init__(
            input_dim, hidden_dims, output_dim, action_space
        )
        if len(hidden_dims) < 1:
            raise ValueError(
                "The hidden dims cannot be empty for a gaussian actor network."
            )

        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            last_activation="relu",
        )
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.fc_std = torch.nn.Linear(hidden_dims[-1], output_dim)
        self._action_space = action_space
        # check this for multi-dimensional spaces
        assert isinstance(action_space, BoxActionSpace)
        self.register_buffer(
            "_action_bound",
            (action_space.high.clone().detach() - action_space.low.clone().detach())
            / 2,
        )

        # preventing the actor network from learning a flat or a point mass distribution
        self._log_std_min = -5
        self._log_std_max = 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._model(x)
        mean = self.fc_mu(x)
        log_std = self.fc_std(x)
        # log_std = torch.clamp(log_std, min=self._log_std_min, max=self._log_std_max)

        # alternate to standard clamping; not sure if it makes a difference but still
        # trying out
        log_std = torch.tanh(log_std)
        log_std = self._log_std_min + 0.5 * (self._log_std_max - self._log_std_min) * (
            log_std + 1
        )
        return mean, log_std

    def sample_action(
        self, state_batch: Tensor, get_log_prob: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample an action from the actor network.

        Args:
            state_batch: A tensor of states.  # TODO: Enforce batch shape?
            get_log_prob: If True, also return the log probability of the sampled actions.

        Returns:
            action: Sampled action, scaled to the action space bounds.
        """
        epsilon = 1e-6
        mean, log_std = self.forward(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)
        sample = normal.rsample()  # reparameterization trick

        # ensure sampled action is within [-1, 1]^{action_dim}
        normalized_action = torch.tanh(sample)

        # clamp each action dimension to prevent numerical issues in tanh
        # normalized_action.clamp(-1 + epsilon, 1 - epsilon)
        action = action_scaling(self._action_space, normalized_action)

        log_prob = normal.log_prob(sample)
        log_prob -= torch.log(
            self._action_bound * (1 - normalized_action.pow(2)) + epsilon
        )

        # for multi-dimensional action space, sum log probabilities over individual
        # action dimension
        if log_prob.dim() == 2:
            log_prob = log_prob.sum(dim=1, keepdim=True)

        if get_log_prob:
            return action, log_prob
        else:
            return action

    def get_log_probability(
        self, state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> Tensor:
        """
        Compute log probability of actions, pi(a|s) under the policy parameterized by
        the actor network.
        Args:
            state_batch: batch of states
            action_batch: batch of actions
        Returns:
            log_prob: log probability of each action in the batch
        """
        epsilon = 1e-6
        mean, log_std = self.forward(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)

        # assume that input actions are in [-1, 1]^d
        # TODO: change this to add a transform for unscaling and uncentering depending on the
        # action space
        unscaled_action_batch = torch.clip(action_batch, -1 + epsilon, 1 - epsilon)

        # transform actions from [-1, 1]^d to [-inf, inf]^d
        unnormalized_action_batch = torch.atanh(unscaled_action_batch)
        log_prob = normal.log_prob(unnormalized_action_batch)
        log_prob -= torch.log(
            self._action_bound * (1 - unscaled_action_batch.pow(2)) + epsilon
        )

        # for multi-dimensional action space, sum log probabilities over individual
        # action dimension
        if log_prob.dim() == 2:
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob
