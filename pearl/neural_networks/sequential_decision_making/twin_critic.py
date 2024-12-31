# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import inspect
from collections.abc import Callable, Iterable
from typing import Optional

import torch
import torch.nn as nn
from pearl.neural_networks.common.utils import init_weights
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
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
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_dims: Optional[Iterable[int]] = None,
        init_fn: Callable[[nn.Module], None] = init_weights,
        network_type: type[QValueNetwork] = VanillaQValueNetwork,
        output_dim: int = 1,
        network_instance_1: Optional[QValueNetwork] = None,
        network_instance_2: Optional[QValueNetwork] = None,
    ) -> None:
        super().__init__()
        if network_instance_1 is not None and network_instance_2 is not None:
            self._critic_1: QValueNetwork = network_instance_1
            self._critic_2: QValueNetwork = network_instance_2
        else:
            assert state_dim is not None
            assert action_dim is not None
            assert hidden_dims is not None
            assert network_type is not None
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_batch (torch.Tensor): a batch of states with shape (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of actions with shape (batch_size,
                number_of_actions_to_query, action_dim)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q-values of (state, action) pairs with shape
            (batch_size)
        """
        critic_1_values = self._critic_1.get_q_values(state_batch, action_batch)
        critic_2_values = self._critic_2.get_q_values(state_batch, action_batch)
        return critic_1_values, critic_2_values
