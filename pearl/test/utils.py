#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
This file contains helpers for unittest creation
"""

from typing import Tuple

import torch


# for testing vanilla mlps
def create_normal_pdf_training_data(
    input_dim: int, num_data_points: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    x are sampled from a multivariate normal distribution
    y are the corresponding pdf values
    """
    mean = torch.zeros(input_dim)
    sigma = torch.eye(input_dim)

    multi_variate_normal = torch.distributions.MultivariateNormal(mean, sigma)
    x = multi_variate_normal.sample(
        torch.Size((num_data_points,))
    )  # sample from a mvn distribution
    y = torch.exp(
        -0.5 * ((x - mean) @ torch.inverse(sigma) * (x - mean)).sum(dim=1)
    ) / (
        # pyre-ignore[58]: `**` is not supported for operand types `Tensor` and `int`.
        # PyTorch idiosyncrasy.
        torch.sqrt((2 * torch.tensor(3.14)) ** mean.shape[0] * torch.det(sigma))
    )  # corresponding pdf of mvn
    y_corrupted = y + 0.01 * torch.randn(num_data_points)  # noise corrupted targets
    return x, y_corrupted
