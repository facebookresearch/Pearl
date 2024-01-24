# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Optional, Tuple

import torch
from pearl.neural_networks.contextual_bandit.base_cb_model import MuSigmaCBModel
from pearl.utils.device import is_distribution_enabled


logger: logging.Logger = logging.getLogger(__name__)


class LinearRegression(MuSigmaCBModel):
    def __init__(self, feature_dim: int, l2_reg_lambda: float = 1.0) -> None:
        """
        A linear regression model which can estimate both point prediction and uncertainty
            (standard delivation).
        Based on the LinUCB paper: https://arxiv.org/pdf/1003.0146.pdf
        Note that instead of being trained by a PyTorch optimizer, we explicitly
            update attributes A and b (according to the LinUCB formulas implemented in
            learn_batch() method)
        An extra column of ones is appended to the input data for the intercept where necessary.
            A user should not append a column of ones to the input data.

        feature_dim: number of features
        l2_reg_lambda: L2 regularization parameter
        """
        super(LinearRegression, self).__init__(feature_dim=feature_dim)
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
        The calculation of pred_sigma (uncertainty) in LinUCB is done by quadratic form x^T * A^{-1} * x.
        Inspired by https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v  # noqa: E501
        This is a vectorized implementation of out[i] = x[i].t() @ A @ x[i]
        x shape: (Batch, Feature_dim)
        A shape: (Feature_dim, Feature_dim)
        output shape: (Batch, 1)
        """
        return (torch.matmul(x, A) * x).sum(-1).unsqueeze(-1)

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
        # pyre-ignore[16]: Module `_C` has no attribute `_LinAlgError`.
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
        self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if weight is None:
            weight = torch.ones_like(y)
        if weight.ndim == 1:
            logger.warning("2D shape expected for weight, got 1D shape {weight.shape}")
            weight = weight.unsqueeze(-1)
        assert x.shape == (
            batch_size,
            self._feature_dim,
        ), f"x has shape {x.shape} != {(batch_size, self._feature_dim)}"
        assert y.shape == (batch_size, 1), f"y has shape {y.shape} != {(batch_size, 1)}"
        assert weight.shape == (
            batch_size,
            1,
        ), f"weight has shape {weight.shape} != {(batch_size, 1)}"
        x = self.append_ones(x)
        return x, y, weight

    def learn_batch(
        self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor]
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
        # x can be [batch_size, feature_dim] or [batch_size, num_arms, feature_dim]
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]

        # dim: [batch_size * num_arms, feature_dim]
        x = x.reshape(-1, feature_dim)

        x = self.append_ones(x)
        # dim: [batch_size, num_arms]
        return torch.matmul(x, self.coefs.t()).reshape(batch_size, -1)

    def calculate_coefs(self) -> None:
        """
        Calculate coefficients based on current A and b.
        Save inverted A and coefficients in buffers.
        """
        self._inv_A = self.matrix_inv_fallback_pinv(self._A)
        self._coefs = torch.matmul(self._inv_A, self._b)

    def calculate_sigma(self, x: torch.Tensor) -> torch.Tensor:
        # x can be [batch_size, feature_dim] or [batch_size, num_arms, feature_dim]
        batch_size = x.shape[0]
        feature_dim = x.shape[-1]

        # dim: [batch_size * num_arms, feature_dim]
        x = x.reshape(-1, feature_dim)
        x = self.append_ones(x)
        sigma = torch.sqrt(self.batch_quadratic_form(x, self._inv_A))
        return sigma.reshape(batch_size, -1)

    def __str__(self) -> str:
        return f"LinearRegression(A:\n{self._A}\nb:\n{self._b})"
