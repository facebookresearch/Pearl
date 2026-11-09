# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Independent continuous-action quantile critics for TQC."""

import torch
from pearl.neural_networks.common.utils import mlp_block
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from torch import nn, Tensor


class QuantileCriticEnsemble(QValueNetwork):
    """Independent critics with output shape [batch, critics, quantiles].

    Each critic receives the same state and action in environment units.
    get_q_values averages all predictions. This network does not implement
    Pearl's distributional safety interface.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        num_critics: int = 2,
        num_quantiles: int = 25,
    ) -> None:
        super().__init__()
        for name, value in [
            ("state_dim", state_dim),
            ("action_dim", action_dim),
            ("num_critics", num_critics),
            ("num_quantiles", num_quantiles),
            *[("hidden_dim", width) for width in hidden_dims],
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._state_dim = state_dim
        self._action_dim = action_dim
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles
        self.critics = nn.ModuleList(
            [
                mlp_block(state_dim + action_dim, hidden_dims, output_dim=num_quantiles)
                for _ in range(num_critics)
            ]
        )
        self.register_buffer(
            "quantile_midpoints", (torch.arange(num_quantiles) + 0.5) / num_quantiles
        )

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def get_q_value_distribution(
        self, state_batch: Tensor, action_batch: Tensor
    ) -> Tensor:
        if state_batch.ndim != 2 or state_batch.shape[-1] != self._state_dim:
            raise ValueError(f"states must have shape [batch, {self._state_dim}]")
        if action_batch.ndim != 2 or action_batch.shape != (
            state_batch.shape[0],
            self._action_dim,
        ):
            raise ValueError(f"actions must have shape [batch, {self._action_dim}]")
        features = torch.cat([state_batch, action_batch], dim=-1)
        return torch.stack([critic(features) for critic in self.critics], dim=1)

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Tensor | None = None,
    ) -> Tensor:
        return self.get_q_value_distribution(state_batch, action_batch).mean(dim=(1, 2))
