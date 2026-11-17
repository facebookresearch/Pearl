# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""Quantile value networks for a fixed, indexed discrete action catalog."""

import torch
from pearl.neural_networks.common.noisy_linear import NoisyLinear
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QuantileQValueNetwork,
)
from torch import nn, Tensor


class RainbowQValueNetwork(QuantileQValueNetwork):
    """A quantile network with optional dueling heads and noisy linear layers.

    This is the quantile variant of Rainbow's network, not its original C51
    categorical network. Actions are scalar indices in ``[0, num_actions)``.
    ``get_q_value_distribution`` accepts ``[B, 1]`` or ``[B, Q, 1]`` actions and
    returns ``[B, N]`` or ``[B, Q, N]`` quantiles respectively.

    The dueling advantage mean is always over the complete fixed action catalog,
    before gathering queried actions. Available-action masks belong to the
    learner's action selection; changing a query subset must not change Z(s, a).
    Vector observations and a fixed catalog are supported; action embeddings and
    dynamically growing catalogs require a different network interface.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        num_quantiles: int,
        *,
        num_actions: int,
        dueling: bool = True,
        noisy: bool = True,
        std_init: float = 0.5,
    ) -> None:
        # Retain QuantileQValueNetwork's interface without constructing its
        # unrelated state-action concatenation network.
        nn.Module.__init__(self)
        if state_dim <= 0 or num_actions <= 0 or num_quantiles <= 0:
            raise ValueError(
                "state_dim, num_actions, and num_quantiles must be positive"
            )
        if action_dim != 1:
            raise ValueError("RainbowQValueNetwork requires scalar action indices")
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden layer dimensions must be positive")

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._num_quantiles = num_quantiles
        self._num_actions = num_actions
        self._dueling = dueling
        self._noise_enabled = True
        self.register_buffer("_quantiles", torch.linspace(0, 1, num_quantiles + 1))
        self.register_buffer(
            "_quantile_midpoints",
            ((self._quantiles[:-1] + self._quantiles[1:]) / 2).reshape(1, 1, -1),
        )

        def linear(input_dim: int, output_dim: int) -> nn.Module:
            if noisy:
                return NoisyLinear(input_dim, output_dim, std_init=std_init)
            return nn.Linear(input_dim, output_dim)

        layers: list[nn.Module] = []
        feature_dim = state_dim
        for width in hidden_dims:
            layers.extend([linear(feature_dim, width), nn.ReLU()])
            feature_dim = width
        self.feature_network = nn.Sequential(*layers)

        if dueling:
            self.value_head = linear(feature_dim, num_quantiles)
            self.advantage_head = linear(feature_dim, num_actions * num_quantiles)
            self.quantile_head = None
        else:
            self.value_head = None
            self.advantage_head = None
            self.quantile_head = linear(feature_dim, num_actions * num_quantiles)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def noise_enabled(self) -> bool:
        return self._noise_enabled

    def enable_noise(self, enabled: bool) -> None:
        self._noise_enabled = enabled
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.enable_noise(enabled)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, state_batch: Tensor) -> Tensor:
        """Return quantiles for every catalog action: ``[B, A, N]``."""
        if state_batch.ndim != 2 or state_batch.shape[-1] != self._state_dim:
            raise ValueError(f"states must have shape [batch, {self._state_dim}]")
        features = self.feature_network(state_batch)
        batch_size = state_batch.shape[0]
        if self._dueling:
            value = self.value_head(features).unsqueeze(1)
            advantage = self.advantage_head(features).reshape(
                batch_size, self._num_actions, self._num_quantiles
            )
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.quantile_head(features).reshape(
            batch_size, self._num_actions, self._num_quantiles
        )

    def get_q_value_distribution(
        self, state_batch: Tensor, action_batch: Tensor
    ) -> Tensor:
        if action_batch.ndim not in (2, 3) or action_batch.shape[-1] != 1:
            raise ValueError(
                "actions must have shape [batch, 1] or [batch, queries, 1]"
            )
        if action_batch.shape[0] != state_batch.shape[0]:
            raise ValueError("state and action batch sizes must match")
        if (
            action_batch.dtype == torch.bool
            or action_batch.is_complex()
            or not torch.isfinite(action_batch).all()
        ):
            raise ValueError("action indices must be finite integers")
        if action_batch.is_floating_point() and not torch.equal(
            action_batch, action_batch.round()
        ):
            raise ValueError("action indices must be integers")
        if torch.any(action_batch < 0) or torch.any(action_batch >= self._num_actions):
            raise ValueError("action index is outside the fixed action catalog")

        single_action = action_batch.ndim == 2
        query_actions = action_batch.unsqueeze(1) if single_action else action_batch
        all_quantiles = self.forward(state_batch)
        indices = query_actions.to(device=all_quantiles.device, dtype=torch.long)
        quantiles = all_quantiles.gather(1, indices.expand(-1, -1, self._num_quantiles))
        return quantiles.squeeze(1) if single_action else quantiles

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Tensor | None = None,
    ) -> Tensor:
        """Expected Q values; availability does not change catalog centering."""
        return self.get_q_value_distribution(state_batch, action_batch).mean(dim=-1)
