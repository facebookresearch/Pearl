# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Layer-normalized Q network for fixed discrete actions.

Architecture follows the vector-control PQN configuration described by CleanRL,
commit fe8d8a03c41a7ef5b523e2e354bd01c363e786bb, cleanrl/pqn.py. This interface
implementation is original; see Gallici et al., arXiv:2407.04811, for the algorithm.
"""

import math

import torch
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from torch import nn


class ParallelQValueNetwork(QValueNetwork):
    """Orthogonally initialized MLP, with LayerNorm before every hidden ReLU."""

    def __init__(
        self, state_dim: int, num_actions: int, hidden_dims: list[int] | None = None
    ) -> None:
        super().__init__()
        hidden_dims = [120, 84] if hidden_dims is None else list(hidden_dims)
        for value in [state_dim, num_actions, *hidden_dims]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("network dimensions must be positive integers")
        self._state_dim, self._num_actions = state_dim, num_actions
        layers: list[nn.Module] = []
        width = state_dim
        for next_width in hidden_dims:
            linear = nn.Linear(width, next_width)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.LayerNorm(next_width), nn.ReLU()])
            width = next_width
        output = nn.Linear(width, num_actions)
        nn.init.orthogonal_(output.weight, gain=math.sqrt(2))
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return 1

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if state_batch.ndim != 2 or state_batch.shape[-1] != self._state_dim:
            raise ValueError(f"states must have shape [batch, {self._state_dim}]")
        return self.network(state_batch)

    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            action_batch.ndim not in (2, 3)
            or action_batch.shape[-1] != 1
            or action_batch.shape[0] != state_batch.shape[0]
        ):
            raise ValueError(
                "actions must have shape [batch, 1] or [batch, queries, 1]"
            )
        if (
            action_batch.dtype == torch.bool
            or action_batch.is_complex()
            or not torch.isfinite(action_batch).all()
            or not torch.equal(action_batch, action_batch.round())
            or (action_batch < 0).any()
            or (action_batch >= self._num_actions).any()
        ):
            raise ValueError("actions must be integer IDs within the fixed catalog")
        values = self.forward(state_batch)
        single = action_batch.ndim == 2
        queries = action_batch if single else action_batch.squeeze(-1)
        result = values.gather(1, queries.to(device=values.device, dtype=torch.long))
        return result.squeeze(-1) if single else result
