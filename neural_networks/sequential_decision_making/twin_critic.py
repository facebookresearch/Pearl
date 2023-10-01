from typing import Callable, List, Tuple, Type

import torch
import torch.nn as nn
from pearl.neural_networks.common.value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from torch import optim


class TwinCritic(torch.nn.Module):
    """
    This is a wrapper for using two critic networks to reduce overestimation bias in q value estimates.
    Each critic network is initialized differently by a given random initialization function.

    NOTE: For more than two critics, the standard way is to use nn.ModuleList()
    """

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: int,
        learning_rate: float,
        network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        # pyre-fixme[9]: init_fn has type `(Module) -> None`; used as `None`.
        init_fn: Callable[[torch.nn.Module], None] = None,
        output_dim: int = 1,
    ):
        super(TwinCritic, self).__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        # pyre-fixme[45]: Cannot instantiate abstract class `QValueNetwork`.
        self._critic_1 = network_type(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        # pyre-fixme[4]: Attribute must be annotated.
        # pyre-fixme[45]: Cannot instantiate abstract class `QValueNetwork`.
        self._critic_2 = network_type(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

        # nn.ModuleList helps manage the networks (initilization, parameter update etc.) efficiently
        self._critic_networks_combined = nn.ModuleList([self._critic_1, self._critic_2])
        self._critic_networks_combined.apply(init_fn)
        self._optimizer = optim.AdamW(
            [
                {"params": self._critic_1.parameters()},
                {"params": self._critic_2.parameters()},
            ],
            lr=learning_rate,
            amsgrad=True,
        )

    def optimize_twin_critics_towards_target(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        expected_target: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Performs an optimization step on the two critic networks towards a given target.
        Args:
            state_batch: a batch of states with shape (batch_size, state_dim)
            action_batch: a batch of actions with shape (batch_size, action_dim)
            expected_target: the batch of target estimates.
        Returns:
            List[torch.Tensor]: individual critic losses along with the mean loss.
        """
        criterion = torch.nn.MSELoss()
        self._optimizer.zero_grad()
        q_1, q_2 = self.get_twin_critic_values(state_batch, action_batch)
        loss = criterion(q_1, expected_target) + criterion(q_2, expected_target)
        loss.backward()
        self._optimizer.step()

        # pyre-fixme[7]: Expected `List[Tensor]` but got `Dict[str, typing.Any]`.
        return {
            "mean_loss": loss.item(),
            "critic_1_loss": q_1.mean().item(),
            "critic_2_loss": q_2.mean().item(),
        }

    def get_twin_critic_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_batch (torch.Tensor): a batch of states with shape (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of actions with shape (batch_size, action_dim)
        Returns:
            torch.Tensor: Q-values of (state, action) pairs with shape (batch_size)
        """
        critic_1_values = self._critic_1.get_q_values(state_batch, action_batch)
        critic_2_values = self._critic_2.get_q_values(state_batch, action_batch)
        return critic_1_values, critic_2_values
