#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Linear Regression Classes
Currently used for linUCB and disjointLinUCB
paper: https://arxiv.org/pdf/1003.0146.pdf

TODO: Distribution and many other production issues need to be considered
Currently only contains simplest logic
Before migrating to production, needs to schedule a code review to compare with ReAgent
fbcode/reagent/models/disjoint_linucb_predictor.py
fbcode/reagent/models/linear_regression.py
"""

import logging
from typing import Tuple

import torch
from pearl.utils.device import is_distribution_enabled
from torch import nn


logger: logging.Logger = logging.getLogger(__name__)


class LinearRegression(nn.Module):
    def __init__(self, feature_dim: int, l2_reg_lambda: float = 1.0) -> None:
        """
        feature_dim: number of features
        l2_reg_lambda: L2 regularization parameter
        """
        super(LinearRegression, self).__init__()
        self.register_buffer(
            "_A",
            l2_reg_lambda * torch.eye(feature_dim + 1),  # +1 for intercept
        )
        self.register_buffer("_b", torch.zeros(feature_dim + 1))
        self.register_buffer("_sum_weight", torch.zeros(1))
        self.register_buffer(
            "_inv_A",
            torch.zeros(feature_dim + 1, feature_dim + 1),
        )
        self.register_buffer("_coefs", torch.zeros(feature_dim + 1))
        self._feature_dim = feature_dim
        self.distribution_enabled: bool = is_distribution_enabled()

    @property
    def A(self) -> torch.Tensor:
        return self._A

    @property
    def coefs(self) -> torch.Tensor:
        return self._coefs

    @staticmethod
    def batch_quadratic_form(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the quadratic form x^T * A * x for a batched input x.
        The calcuation of pred_sigma (uncertainty) in LinUCB is done by quadratic form x^T * A^{-1} * x.
        Inspired by https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v
        This is a vectorized implementation of out[i] = x[i].t() @ A @ x[i]
        x shape: (Batch, Feature_dim)
        A shape: (Feature_dim, Feature_dim)
        output shape: (Batch)
        """
        return (torch.matmul(x, A) * x).sum(-1)

    @staticmethod
    def append_ones(x: torch.Tensor) -> torch.Tensor:
        """
        Append a column of ones to x (for intercept of linear regression)
        We append at the beginning along the last dimension (features)
        """
        # Create a tensor of ones to append
        ones = torch.ones_like(torch.select(x, dim=-1, index=0).unsqueeze(-1))

        # Concatenate the input data with the tensor of ones along the last dimension
        result = torch.cat((ones, x), dim=-1)

        return result

    @staticmethod
    def matrix_inv_fallback_pinv(A: torch.Tensor) -> torch.Tensor:
        """
        Try to apply regular matrix inv. If it fails, fallback to pseudo inverse
        """
        try:
            inv_A = torch.linalg.inv(A).contiguous()
        # pyre-fixme[16]: Module `_C` has no attribute `_LinAlgError`.
        except torch._C._LinAlgError as e:
            logger.warning(
                "Exception raised during A inversion, falling back to pseudo-inverse",
                e,
            )
            # switch from `inv` to `pinv`
            # first check if A is Hermitian (symmetric A)
            A_is_hermitian = torch.allclose(A, A.T, atol=1e-4, rtol=1e-4)
            # applying hermitian=True saves about 50% computations
            inv_A = torch.linalg.pinv(
                A,
                hermitian=A_is_hermitian,
            ).contiguous()
        return inv_A

    def _validate_train_inputs(
        self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        if weight is None:
            weight = torch.ones_like(y)
        assert x.shape == (
            batch_size,
            self._feature_dim,
        ), f"x has shape {x.shape} != {(batch_size, self._feature_dim)}"
        assert y.shape == (batch_size,), f"y has shape {y.shape} != {(batch_size,)}"
        assert weight.shape == (
            batch_size,
        ), f"weight has shape {weight.shape} != {(batch_size,)}"
        y = torch.unsqueeze(y, dim=1)
        weight = torch.unsqueeze(weight, dim=1)
        x = self.append_ones(x)
        return x, y, weight

    def learn_batch(
        self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
    ) -> None:
        """
        A <- A + x*x.t
        b <- b + r*x
        """
        # this also appends a column of ones to `x`
        x, y, weight = self._validate_train_inputs(x, y, weight)

        delta_A = torch.matmul(x.t(), x * weight)
        delta_b = torch.matmul(x.t(), y * weight).squeeze()
        delta_sum_weight = weight.sum()

        if self.distribution_enabled:
            torch.distributed.all_reduce(delta_A)
            torch.distributed.all_reduce(delta_b)
            torch.distributed.all_reduce(delta_sum_weight)

        self._A += delta_A.to(self._A.device)
        self._b += delta_b.to(self._b.device)
        self._sum_weight += delta_sum_weight.to(self._sum_weight.device)

        self.calculate_coefs()  # update coefs after updating A and b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x could be a single vector or a batch
        If x is a batch, it will be shape(batch_size, ...)
        return will be shape(batch_size)
        """
        x = self.append_ones(x)
        return torch.matmul(x, self._coefs.t())

    def calculate_coefs(self) -> None:
        """
        Calculate coefficients based on current A and b.
        Save inverted A and coefficients in buffers.
        """
        self._inv_A = self.matrix_inv_fallback_pinv(self._A)
        self._coefs = torch.matmul(self._inv_A, self._b)

    def calculate_sigma(self, x: torch.Tensor) -> torch.Tensor:
        x = self.append_ones(x)  # append a column of ones for intercept
        sigma = torch.sqrt(self.batch_quadratic_form(x, self._inv_A))
        return sigma

    def __str__(self) -> str:
        return f"LinearRegression(A:\n{self._A}\nb:\n{self._b})"
