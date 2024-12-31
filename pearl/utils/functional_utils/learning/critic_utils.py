# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn

from pearl.neural_networks.common.utils import (
    init_weights,
    update_target_network,
    update_target_networks,
)
from pearl.neural_networks.common.value_networks import (
    ValueNetwork,
    VanillaValueNetwork,
)

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic

"""
This file is a collection of some functions used to create and update critic networks
as well as compute optimization losses.
"""
# TODO 1: see if we can remove the `update_critic_target_networks` and
# `single_critic_state_value_loss` functions.

# TODO 2: see if we can add functions for updating the target networks and computing losses
# in the `TwinCritic` class.


def make_critic(
    state_dim: int,
    hidden_dims: Iterable[int] | None,
    use_twin_critic: bool,
    network_type: type[ValueNetwork] | type[QValueNetwork],
    action_dim: int | None = None,
) -> nn.Module:
    """
    A utility function to instantiate a critic network. Critic networks are used by different
    modules in Pearl, for example, in different actor-critic algorithms, reward constrained
    safety module etc.

    Args:
        state_dim (int): Dimension of the observation space.
        hidden_dims (Optional[Iterable[int]]): Hidden dimensions of the critic network.
        use_twin_critic (bool): Whether to use a twin critic or not. If set to True, an object of
            class `TwinCritic` will be instantiated with the specified `network_type`. Note that
            setting `use_twin_critic` = True is only supported for specified `network_type`
            to be a subclass of `QValueNetwork`.
        network_type (Union[Type[ValueNetwork], Type[QValueNetwork]]): The type of the critic
            network to instantiate.
        action_dim (Optional[int]): The dimension of the action space.

    Returns:
        critic network (nn.Module): The critic network to be used by different modules.
    """
    if use_twin_critic:
        assert action_dim is not None
        assert hidden_dims is not None
        assert issubclass(
            network_type, QValueNetwork
        ), "network_type must be a subclass of QValueNetwork when use_twin_critic is True"

        # cast network_type to get around static Pyre type checking; the runtime check with
        # `issubclass` ensures the network type is a sublcass of QValueNetwork
        # pyre-fixme[22]: The cast is redundant.
        network_type = cast(type[QValueNetwork], network_type)

        return TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            network_type=network_type,
            init_fn=init_weights,
        )
    else:
        if network_type == VanillaQValueNetwork:
            # pyre-ignore[45]:
            # Pyre does not know that `network_type` is asserted to be concrete
            return network_type(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )
        elif network_type == VanillaValueNetwork:
            # pyre-ignore[45]:
            # Pyre does not know that `network_type` is asserted to be concrete
            return network_type(
                input_dim=state_dim, hidden_dims=hidden_dims, output_dim=1
            )
        else:
            raise NotImplementedError(
                f"Type {network_type} cannot be used to instantiate a critic network."
            )


def update_critic_target_network(
    target_network: nn.Module, network: nn.Module, tau: float
) -> None:
    """
    Updates the target network of a critic network.

    Args:
        target_network (nn.Module): The target network to update.
        network (nn.Module): The source network to update the wieghts of the target network from.
        tau (float): Coefficient for soft updates.
    """
    # if twin critics are used, update both critic networks of the twin critic object
    if isinstance(target_network, TwinCritic):
        update_target_networks(
            target_network._critic_networks_combined,
            # pyre-fixme[6]: For 2nd argument expected `Union[List[Module],
            #  ModuleList]` but got `Union[Module, Tensor]`.
            network._critic_networks_combined,
            tau=tau,
        )
    else:
        update_target_network(
            (
                # pyre-fixme[6]: For 1st argument expected `Module` but got
                #  `Union[Module, Tensor]`.
                target_network._model
                if hasattr(target_network, "_model")
                else target_network
            ),
            # pyre-fixme[6]: For 2nd argument expected `Module` but got
            #  `Union[Module, Tensor]`.
            network._model if hasattr(network, "_model") else network,
            tau=tau,
        )


def single_critic_state_value_loss(
    state_batch: torch.Tensor,
    expected_target_batch: torch.Tensor,
    critic: nn.Module,
) -> torch.Tensor:
    """
    This method calculates the mean squared error between the predicted state values from the
    critic network and the input target estimates.

    Args:
        state_batch (torch.Tensor): A batch of states with expected shape `(batch_size, state_dim)`.
        expected_target_batch (torch.Tensor): The batch of target estimates (i.e., RHS of the
            Bellman equation) with shape `(batch_size)`.
        critic (nn.Module): The critic network to update.
    Returns:
        loss (torch.Tensor): Mean squared error in the Bellman equation for state-value prediction
        with expected shape as `()`. This scalar loss is used to train the value critic network.
    """
    if not isinstance(critic, ValueNetwork):
        raise TypeError(
            "critic in the `single_critic_state_value_update` method must be an instance of "
            "ValueNetwork"
        )
    vs = critic(state_batch)
    criterion = torch.nn.MSELoss()
    loss = criterion(
        vs.reshape_as(expected_target_batch), expected_target_batch.detach()
    )
    return loss


def twin_critic_action_value_loss(
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    expected_target_batch: torch.Tensor,
    critic: TwinCritic,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This method calculates the sum of the mean squared errors between the predicted Q-values
    using critic networks (LHS of the Bellman equation) and the input target estimates (RHS of the
    Bellman equation).

    Args:
        state_batch (torch.Tensor): A batch of states with expected shape
            `(batch_size, state_dim)`.
        action_batch (torch.Tensor): A batch of actions with expected shape
            `(batch_size, action_dim)`.
        expected_target_batch (torch.Tensor): The batch of target estimates
            (i.e. RHS of the Bellman equation) with expected shape `(batch_size)`.
        critic (TwinCritic): The twin critic network to update.
    Returns:
        loss (torch.Tensor): Sum of mean squared errors in the Bellman equation (for action-value
            prediction) corresponding to both critic networks. The expected shape is `()`. This
            scalar loss is used to train both critics of the twin critic network.
        q1: q1 critic network prediction
        q2: q2 critic network prediction
    """

    criterion = torch.nn.MSELoss()
    q_1, q_2 = critic.get_q_values(state_batch, action_batch)
    loss = criterion(
        q_1.reshape_as(expected_target_batch), expected_target_batch.detach()
    ) + criterion(q_2.reshape_as(expected_target_batch), expected_target_batch.detach())
    loss = loss / 2.0
    return loss, q_1, q_2
