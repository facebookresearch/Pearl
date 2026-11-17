# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""Factorized Gaussian parameter noise for value-based exploration."""

import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """A linear layer with learned noise scales and factorized Gaussian noise.

    Noise is held fixed between calls to ``reset_noise``. Evaluation mode or
    ``enable_noise(False)`` uses the learned means. The explicit noise switch,
    like ``Module.training``, is an execution mode rather than checkpoint state;
    learned means/scales and the current noise samples are in ``state_dict``.

    See Fortunato et al., *Noisy Networks for Exploration*, arXiv:1706.10295.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        std_init: float = 0.5,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if not math.isfinite(std_init) or std_init < 0:
            raise ValueError("std_init must be finite and nonnegative")

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self._noise_enabled = True

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer("bias_epsilon", torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_sigma", None)
            self.register_buffer("bias_epsilon", None)

        self.reset_parameters()
        self.reset_noise()

    @property
    def noise_enabled(self) -> bool:
        return self._noise_enabled

    def enable_noise(self, enabled: bool) -> None:
        self._noise_enabled = enabled

    @torch.no_grad()
    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.uniform_(-bound, bound)
        self.weight_sigma.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.uniform_(-bound, bound)
            self.bias_sigma.fill_(self.std_init / math.sqrt(self.out_features))

    @torch.no_grad()
    def reset_noise(self) -> None:
        def transformed_normal(size: int) -> Tensor:
            noise = self.weight_mu.new_empty(size).normal_()
            return noise.sign() * noise.abs().sqrt()

        input_noise = transformed_normal(self.in_features)
        output_noise = transformed_normal(self.out_features)
        self.weight_epsilon.copy_(output_noise.outer(input_noise))
        if self.bias_epsilon is not None:
            self.bias_epsilon.copy_(output_noise)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training and self._noise_enabled:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = (
                None
                if self.bias_mu is None
                else self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(inputs, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias_mu is not None}, std_init={self.std_init}"
        )
