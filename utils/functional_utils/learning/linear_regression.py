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
from pearl.neural_networks.common.auto_device_nn_module import AutoDeviceNNModule
from pearl.utils.device import get_pearl_device

logger = logging.getLogger(__name__)


class LinearRegression(AutoDeviceNNModule):
    def __init__(self, feature_dim: int) -> None:
        super(LinearRegression, self).__init__()
        self.device = get_pearl_device()
        self.register_buffer("_A", 1e-2 * torch.eye(feature_dim, device=self.device))
        self.register_buffer("_b", torch.zeros(feature_dim, device=self.device))
        self._feature_dim = feature_dim

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
        return x, y, weight

    def learn_batch(
        self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
    ) -> None:
        """
        A <- A + x*x.t
        b <- b + r*x
        """
        x, y, weight = self._validate_train_inputs(x, y, weight)
        delta_A = torch.matmul(x.t(), x * weight)
        delta_b = torch.matmul(x.t(), y * weight).squeeze()
        if self.distribution_enabled:
            torch.distributed.all_reduce(delta_A)
            torch.distributed.all_reduce(delta_b)
        self._A += delta_A.to(self._A.device)
        self._b += delta_b.to(self._b.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x could be a single vector or a batch
        If x is a batch, it will be shape(batch_size, ...)
        return will be shape(batch_size)
        """
        return torch.matmul(x, self.coefs.t())

    @property
    def coefs(self) -> torch.Tensor:
        inv_A = self.inv_A
        assert inv_A.size()[0] == self._b.size()[0]
        return torch.matmul(inv_A, self._b)

    @property
    def inv_A(self) -> torch.Tensor:
        return self._A_inv_fallback_pinv()

    def _A_inv_fallback_pinv(self) -> torch.Tensor:
        """
        if the A A or any A in the batch of matrices A is not invertible,
        raise a error message and then switch from `inv` to `pinv`
        https://pytorch.org/docs/stable/generated/torch.linalg.inv.html
        """
        try:
            inv_A = torch.linalg.inv(self._A).contiguous()
        except RuntimeError as e:
            logger.warning(
                "Exception raised during A inversion, falling back to pseudo-inverse",
                e,
            )
            # switch from `inv` to `pinv`
            # first check if A is Hermitian (symmetric A)
            A_is_hermitian = torch.allclose(self._A, self._A.T, atol=1e-4, rtol=1e-4)
            # applying hermitian=True saves about 50% computations
            inv_A = torch.linalg.pinv(
                self._A,
                hermitian=A_is_hermitian,
            ).contiguous()
        return inv_A

    def __str__(self) -> str:
        return f"A:\n{self._A}\nb:\n{self._b}"


class AvgWeightLinearRegression(LinearRegression):
    def __init__(
        self,
        feature_dim: int,
    ) -> None:
        super(AvgWeightLinearRegression, self).__init__(feature_dim=feature_dim)
        # initialize sum of weights below at small values to avoid dividing by 0
        self._sum_weight = 1e-5 * torch.ones(1, dtype=torch.float, device=self.device)

    @property
    def sum_weight(self) -> torch.Tensor:
        return self._sum_weight

    def learn_batch(
        self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
    ) -> None:
        x, y, weight = self._validate_train_inputs(x, y, weight)

        batch_sum_weight = weight.sum()
        self._sum_weight += batch_sum_weight
        # update average values of A and b using observations from the batch
        self._A = (
            self._A * (1 - batch_sum_weight / self._sum_weight)
            + torch.matmul(x.t(), x * weight) / self._sum_weight
        )  # dim (DA*DC, DA*DC)
        self._b = (
            self._b * (1 - batch_sum_weight / self._sum_weight)
            + torch.matmul(x.t(), y * weight).squeeze() / self._sum_weight
        )  # dim (DA*DC,)
