from typing import List, Optional

import torch
from pearl.core.common.neural_networks.utils import update_target_network
from pearl.core.common.neural_networks.value_networks import (
    VanillaStateActionValueNetwork,
)
from torch import optim


class NpletsCritic(torch.nn.Module):
    """
    This is a wrapper for N same critic networks to be used at same time.

    For better Q(s, a) estimation, it is quite common to use 2 same Q network
    to train together with different init. And we use min as an estimation for
    Q(s, a)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims,
        learning_rate,
        num_critics=2,
        network_type=VanillaStateActionValueNetwork,
        init_fn=None,
        output_dim=1,
    ):
        self._critics = []
        self._critics_target = []
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

    def optimize(self, target_fn, expected_target):
        """
        Args:
        target_fn:
            A function with input as one critic network, output is estimated value for expected_target
        expected_target:
            For N Q(s, a), expected target is the same for all critics
        """
        for i, critic in enumerate(self._critics):
            criterion = torch.nn.MSELoss()
            loss = criterion(target_fn(critic), expected_target)
            self._optimizers[i].zero_grad()
            loss.backward()
            self._optimizers[i].step()

    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch: Optional[torch.Tensor] = None,
        target: bool = False,
    ):
        """
        Args
            batch of state: (batch_size, state_dim)
            batch of action: (batch_size, action_dim)
            curr_available_actions_batch is set to None for vanilla state action value network; and is only used
            in the Duelling architecture (see there for more details)
        Return
            q values of (state, action) pairs: (batch_size)
        """
        child_values = []
        critics = self._critics_target if target else self._critics
        for critic in critics:
            child_values.append(
                critic.get_batch_action_value(
                    state_batch, action_batch, curr_available_actions_batch
                )
            )
        return self._apply_min(child_values)

    def _apply_min(self, child_values: List[torch.Tensor]) -> torch.Tensor:
        min_value = child_values[0]
        for child in child_values[1:]:
            min_value = torch.min(min_value, child)
        return min_value

    def update_target_network(self, tau):
        for i, critic in enumerate(self._critics):
            update_target_network(self._critics_target[i], critic, tau)

    def forward(self, x: torch.Tensor, target: bool = False) -> torch.Tensor:
        child_values = []
        critics = self._critics_target if target else self._critics
        for critic in critics:
            child_values.append(critic(x))
        return self._apply_min(child_values)


class TwinCritic(NpletsCritic):
    """
    This is a wrapper for 2 critic networks to be used at same time.
    This is a common trick to apply in algorithms like SAC, TD3 etc.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims,
        learning_rate,
        network_type=VanillaStateActionValueNetwork,
        init_fn=None,
        output_dim=1,
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
