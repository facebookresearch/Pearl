"""
This file is to define different neural network type
And helper API to build different network based on type
"""
from enum import Enum
from typing import List

import torch.nn as nn

from pearl.neural_networks.value_networks import (
    DuelingValueNetwork,
    VanillaValueNetwork,
)


class NetworkType(Enum):
    VANILLA = 0
    DUELING_DQN = 1


def network_maker(
    state_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    network_type: NetworkType,
) -> nn.Module:
    if network_type == NetworkType.VANILLA:
        return VanillaValueNetwork(state_dim + action_dim, hidden_dims, output_dim)
    elif network_type == NetworkType.DUELING_DQN:
        return DuelingValueNetwork(state_dim, action_dim, hidden_dims, output_dim)
    raise NotImplementedError(
        f"network_type={network_type} is not implemented in network_maker"
    )
