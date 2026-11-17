# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unsquashed diagonal Gaussian actor for clipped-action continuous PPO/RPO.

The executed Box action is clipped, but the Gaussian likelihood is evaluated at
the original latent sample. See Schulman et al., arXiv:1707.06347 and Rahman and
Xue, arXiv:2212.07536. This is distinct from SAC's tanh-squashed Gaussian.
"""

import math

import torch
from pearl.neural_networks.common.value_networks import ValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import ActorNetwork
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from torch import nn
from torch.distributions import Normal


def _tanh_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    widths = [input_dim, *hidden_dims, output_dim]
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in widths):
        raise ValueError("network dimensions must be positive integers")
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _orthogonal(module: nn.Sequential, output_gain: float) -> None:
    linear = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for index, layer in enumerate(linear):
        nn.init.orthogonal_(
            layer.weight, output_gain if index == len(linear) - 1 else math.sqrt(2)
        )
        nn.init.zeros_(layer.bias)


class GaussianPPOActorNetwork(ActorNetwork):
    """State-dependent mean and learned state-independent log standard deviation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        action_space: BoxActionSpace,
    ) -> None:
        super().__init__(input_dim, hidden_dims, output_dim, action_space)
        if (
            not isinstance(action_space, BoxActionSpace)
            or action_space.action_dim != output_dim
        ):
            raise ValueError("actor requires a matching Box action space")
        low, high = action_space.low, action_space.high
        if (
            not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or (low >= high).any()
        ):
            raise ValueError("finite strictly increasing action bounds required")
        self.mean_network = _tanh_mlp(input_dim, hidden_dims, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        self.register_buffer("action_low", low.detach().float().clone())
        self.register_buffer("action_high", high.detach().float().clone())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _orthogonal(self.mean_network, 0.01)
        with torch.no_grad():
            self.log_std.zero_()

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.mean_network(state_batch)

    def distribution(
        self, state_batch: torch.Tensor, perturbation_alpha: float = 0.0
    ) -> Normal:
        mean = self.forward(state_batch)
        if perturbation_alpha:
            mean = mean + torch.empty_like(mean).uniform_(
                -perturbation_alpha, perturbation_alpha
            )
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def clip_action(self, latent_action: torch.Tensor) -> torch.Tensor:
        return latent_action.clamp(self.action_low, self.action_high)

    def sample_action(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.clip_action(self.distribution(state_batch).sample())

    def get_action_log_prob(
        self,
        state_batch: torch.Tensor,
        latent_action: torch.Tensor,
        perturbation_alpha: float = 0.0,
    ) -> torch.Tensor:
        return (
            self.distribution(state_batch, perturbation_alpha)
            .log_prob(latent_action)
            .sum(-1)
        )


class PPOValueNetwork(ValueNetwork):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.model = _tanh_mlp(input_dim, hidden_dims, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _orthogonal(self.model, 1.0)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.model(state_batch)
