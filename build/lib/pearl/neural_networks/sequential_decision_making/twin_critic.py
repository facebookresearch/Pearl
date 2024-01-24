# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import inspect
from typing import Callable, Iterable, Tuple, Type

import torch
import torch.nn as nn
from pearl.neural_networks.common.value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)


class TwinCritic(torch.nn.Module):
    """
    This is a wrapper for using two critic networks to reduce overestimation bias in
    critic estimation. Each critic is initialized differently by a given
    initialization function.

    NOTE: For more than two critics, the standard way is to use nn.ModuleList()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        init_fn: Callable[[nn.Module], None],
        network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        output_dim: int = 1,
    ) -> None:
        super(TwinCritic, self).__init__()

        if inspect.isabstract(network_type):
            raise ValueError("network_type must not be abstract")

        # pyre-ignore[45]:
        # Pyre does not know that `network_type` is asserted to be concrete
        self._critic_1: QValueNetwork = network_type(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

        # pyre-ignore[45]:
        # Pyre does not know that `network_type` is asserted to be concrete
        self._critic_2: QValueNetwork = network_type(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

        # nn.ModuleList helps manage the networks
        # (initialization, parameter update etc.) efficiently
        self._critic_networks_combined = nn.ModuleList([self._critic_1, self._critic_2])
        self._critic_networks_combined.apply(init_fn)

    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_batch (torch.Tensor): a batch of states with shape (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of actions with shape (batch_size, action_dim)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q-values of (state, action) pairs with shape
            (batch_size)
        """
        critic_1_values = self._critic_1.get_q_values(state_batch, action_batch)
        critic_2_values = self._critic_2.get_q_values(state_batch, action_batch)
        return critic_1_values, critic_2_values
