# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Categorical values on a fixed support, with optional dueling and NoisyNet."""

import math

import torch
from pearl.neural_networks.common.noisy_linear import NoisyLinear
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from torch import nn, Tensor


class CategoricalRainbowQValueNetwork(QValueNetwork):
    """C51 logits for scalar action IDs, queried through PEARL's expected-Q API.

    Unlike quantile locations, categorical probabilities must be weighted by
    their support to obtain expected returns. Dueling centering is over the
    complete catalog, independently of the queried available-action subset.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: list[int],
        num_atoms: int = 51,
        value_min: float = -10.0,
        value_max: float = 10.0,
        dueling: bool = True,
        noisy: bool = True,
        std_init: float = 0.5,
    ) -> None:
        super().__init__()
        for name, value in [
            ("state_dim", state_dim),
            ("num_actions", num_actions),
            ("num_atoms", num_atoms),
            *[("hidden_dim", n) for n in hidden_dims],
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_atoms < 2:
            raise ValueError("C51 requires at least two support atoms")
        if not (
            math.isfinite(value_min)
            and math.isfinite(value_max)
            and value_min < value_max
        ):
            raise ValueError("support bounds must be finite and strictly increasing")
        if not math.isfinite(std_init) or std_init < 0:
            raise ValueError("std_init must be finite and nonnegative")
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._num_atoms = num_atoms
        self._dueling = dueling
        self._noise_enabled = True
        self.register_buffer("support", torch.linspace(value_min, value_max, num_atoms))

        def linear(inputs: int, outputs: int) -> nn.Module:
            return (
                NoisyLinear(inputs, outputs, std_init=std_init)
                if noisy
                else nn.Linear(inputs, outputs)
            )

        layers: list[nn.Module] = []
        feature_dim = state_dim
        for width in hidden_dims:
            layers.extend([linear(feature_dim, width), nn.ReLU()])
            feature_dim = width
        self.feature_network = nn.Sequential(*layers)
        self.value_head = linear(feature_dim, num_atoms) if dueling else None
        self.advantage_head = (
            linear(feature_dim, num_actions * num_atoms) if dueling else None
        )
        self.categorical_head = (
            None if dueling else linear(feature_dim, num_actions * num_atoms)
        )

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return 1

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
        """Return logits with shape [batch, catalog actions, atoms]."""
        if state_batch.ndim != 2 or state_batch.shape[-1] != self._state_dim:
            raise ValueError(f"states must have shape [batch, {self._state_dim}]")
        features = self.feature_network(state_batch)
        shape = (state_batch.shape[0], self._num_actions, self._num_atoms)
        if self._dueling:
            advantage = self.advantage_head(features).reshape(shape)
            return (
                self.value_head(features).unsqueeze(1)
                + advantage
                - advantage.mean(1, keepdim=True)
            )
        return self.categorical_head(features).reshape(shape)

    def get_logits(self, state_batch: Tensor, action_batch: Tensor) -> Tensor:
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
            or (
                action_batch.is_floating_point()
                and not torch.equal(action_batch, action_batch.round())
            )
            or (action_batch < 0).any()
            or (action_batch >= self._num_actions).any()
        ):
            raise ValueError("actions must be integer IDs within the fixed catalog")
        single = action_batch.ndim == 2
        queries = action_batch.unsqueeze(1) if single else action_batch
        logits = self.forward(state_batch)
        indices = queries.to(device=logits.device, dtype=torch.long)
        result = logits.gather(1, indices.expand(-1, -1, self._num_atoms))
        return result.squeeze(1) if single else result

    def get_probabilities(self, state_batch: Tensor, action_batch: Tensor) -> Tensor:
        return self.get_logits(state_batch, action_batch).softmax(dim=-1)

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Tensor | None = None,
    ) -> Tensor:
        return (self.get_probabilities(state_batch, action_batch) * self.support).sum(
            dim=-1
        )
