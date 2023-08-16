from typing import Callable, List, Optional

import torch
from pearl.core.common.neural_networks.utils import update_target_network
from pearl.core.common.neural_networks.value_networks import (
    StateActionValueNetwork,
    StateActionValueNetworkType,
    VanillaStateActionValueNetwork,
)
from torch import optim


class NpletsCritic(torch.nn.Module):
    """
    This is a wrapper for N critic networks to be used jointly
    as a way to improve Q(s, a) estimates by avoiding an overestimation bias.

    Each critic network is initialized differently by a given random initialization function.

    Each critic network has a corresponding _target_ critical network
    with identical structure and initial parameters.
    The target critical networks are updated less often
    (by the client calling `update_target_networks`)
    than the main critical networks as a way to improve stability.

    When requesting an estimate, the client can specify it to come from either
    the critic networks, or the target critic networks.
    The default is the critic networks.

    The estimate of Q(s, a) is defined as the minimum of the estimates
    from each of the critic networks being used.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: int,
        learning_rate: float,
        num_critics: int = 2,
        network_type: StateActionValueNetworkType = VanillaStateActionValueNetwork,
        init_fn: Callable[[torch.nn.Module], None] = None,
        output_dim: int = 1,
    ):
        self._critics: List[StateActionValueNetwork] = []
        self._critics_target: List[StateActionValueNetwork] = []
        self._optimizers = []
        for i in range(num_critics):
            self._critics.append(
                network_type(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                )
            )
            if init_fn is not None:
                self._critics[i].apply(init_fn)
            self._critics_target.append(
                network_type(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                )
            )
            self._critics_target[i].load_state_dict(self._critics[i].state_dict())
            self._optimizers.append(
                optim.AdamW(
                    self._critics[i].parameters(), lr=learning_rate, amsgrad=True
                )
            )

    def optimize(
        self,
        get_q_value_estimate_fn: Callable[[StateActionValueNetwork], torch.Tensor],
        expected_target: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Performs an optimization step on the N critic networks towards a given target.
        Args:
            get_q_value_estimate_fn: A function evaluating a critic network to obtain a batch of Q-value estimates.
            expected_target: the batch of target Q-value estimates (used for all N critic networks).
        Returns:
            List[torch.Tensor]: The list of mean (over the batch) losses for each of the N critic networks.
        """
        loss_value: List[torch.Tensor] = []
        for i, critic in enumerate(self._critics):
            criterion = torch.nn.MSELoss()
            loss = criterion(get_q_value_estimate_fn(critic), expected_target)
            self._optimizers[i].zero_grad()
            loss.backward()
            self._optimizers[i].step()
            loss_value.append(loss.mean().item())
        return loss_value

    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch: Optional[torch.Tensor] = None,
        target: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            state_batch (torch.Tensor): a batch of state tensors (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of action tensors (batch_size, action_dim)
            curr_available_actions_batch (torch.Tensor, optional): a batch of currently available actions (batch_size, available_action_space_size, action_dim)
            target (bool): whether use the target networks or the main networks.
        Returns:
            torch.Tensor: Q-values of (state, action) pairs: (batch_size)
        """
        critics = self._critics_target if target else self._critics
        list_of_batch_of_q_values_from_each_critic = [
            critic.get_batch_action_value(
                state_batch, action_batch, curr_available_actions_batch
            )
            for critic in critics
        ]
        return torch.stack(list_of_batch_of_q_values_from_each_critic).min(dim=0).values

    def update_target_networks(self, tau: float) -> None:
        for i, critic in enumerate(self._critics):
            update_target_network(self._critics_target[i], critic, tau)

    def forward(self, x: torch.Tensor, target: bool = False) -> torch.Tensor:
        critics = self._critics_target if target else self._critics
        list_of_batch_of_q_values_from_each_critic = [critic(x) for critic in critics]
        return torch.stack(list_of_batch_of_q_values_from_each_critic).min(dim=0).values


class TwinCritic(NpletsCritic):
    """
    An instantiation of NpletsCritic for two critic networks.
    This is a common trick to apply in algorithms like SAC, TD3 etc.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: int,
        learning_rate: float,
        network_type: StateActionValueNetworkType = VanillaStateActionValueNetwork,
        init_fn: Callable[[torch.nn.Module], None] = None,
        output_dim: int = 1,
    ):
        super(TwinCritic, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            learning_rate=learning_rate,
            network_type=network_type,
            init_fn=init_fn,
            num_critics=2,
        )
