# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import logging
from typing import List

import torch
from pearl.neural_networks.contextual_bandit.base_cb_model import MuSigmaCBModel
from pearl.utils.device import is_distribution_enabled


logger: logging.Logger = logging.getLogger(__name__)


class LinearRegression(MuSigmaCBModel):
    def __init__(
        self,
        feature_dim: int,
        l2_reg_lambda: float = 1.0,
        gamma: float = 1.0,
        force_pinv: bool = False,
    ) -> None:
        """
        A linear regression model which can estimate both point prediction and uncertainty
            (standard deviation).
        Based on the LinUCB paper: https://arxiv.org/pdf/1003.0146.pdf

        Note that instead of being trained by a PyTorch optimizer,
        we use the analytical Weight Least Square solution to update the model parameters,
        where the regression coefficients are updated in closed form:
        coefs = (X^T * X)^-1 * X^T * W * y
        where W is an optional weight tensor (e.g. for weighted least squares).
        To compute coefficients, we maintain matrix A = X^T * X and vector b = X^T * W * y,
        which are updated as new data comes in.

        An extra column of ones is appended to the input data for the intercept where necessary.
        A user should not append a column of ones to the input data.

        It furthermore allows for _discounting_. This provides the model with the ability
        to "forget" old data and adjust to a new data distribution in a non-stationary
        environment. The discounting is applied periodically and consists of multiplying
        the underlying linear system matrices A and b (the model's weights) by gamma
        (the discounting multiplier). The discounting period is controlled by
        apply_discounting_interval, which consists of the number of inputs to be
        processed between different rounds of discounting. Note that, because inputs
        are weighted,  apply_discounting_interval is more precisely described as
        the sum of weights of inputs that need to be processed before
        discounting takes place again. This is expressed in pseudo-code as
        ```
        if apply_discounting_interval > 0 and (
                sum_weights - sum_weights_when_last_discounted
                >= apply_discounting_interval:
            A *= discount factor
            b *= discount factor
        ```
        To disable discounting, simply set gamma to 1.

        feature_dim: number of features
        l2_reg_lambda: L2 regularization parameter
        gamma: discounting multiplier (A and b get multiplied by gamma periodically, the period
            is controlled by PolicyLearner). We use a simplified implementation of
            https://arxiv.org/pdf/1909.09146.pdf
        force_pinv: If True, we will always use pseudo inverse to invert the `A` matrix. If False,
            we will first try to use regular matrix inversion. If it fails, we will fallback to
            pseudo inverse.
        """
        super().__init__(feature_dim=feature_dim)
        self.gamma = gamma
        self.l2_reg_lambda = l2_reg_lambda
        self.force_pinv = force_pinv
        assert (
            gamma > 0 and gamma <= 1
        ), f"gamma should be in (0, 1]. Got gamma={gamma} instead"
        self.register_buffer(
            "_A",
            torch.zeros(feature_dim + 1, feature_dim + 1),  # +1 for intercept
        )  # initializing as zeros. L2 regularization will be applied separately.
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
        # return A with L2 regularization applied
        return self._A + self.l2_reg_lambda * torch.eye(
            self._feature_dim + 1, device=self._A.device
        )

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
    def pinv(A: torch.Tensor) -> torch.Tensor:
        """
        Compute the pseudo inverse of A using torch.linalg.pinv
        """
        # first check if A is Hermitian (symmetric A)
        A_is_hermitian = torch.allclose(A, A.T, atol=1e-4, rtol=1e-4)
        # applying hermitian=True saves about 50% computations
        return torch.linalg.pinv(
            A,
            hermitian=A_is_hermitian,
        ).contiguous()

    def matrix_inv_fallback_pinv(self, A: torch.Tensor) -> torch.Tensor:
        """
        Try to apply regular matrix inv. If it fails, fallback to pseudo inverse
        """
        if self.force_pinv:
            return self.pinv(A)
        try:
            return torch.linalg.inv(A).contiguous()
        # pyre-ignore[16]: Module `_C` has no attribute `_LinAlgError`.
        # pyre-fixme[66]: Exception handler type annotation `unknown` must extend
        #  BaseException.
        except torch._C._LinAlgError as e:
            logger.warning(
                "Exception raised during A inversion, falling back to pseudo-inverse",
                e,
            )
            # switch from `inv` to `pinv`
            return self.pinv(A)

    def _validate_train_inputs(
        self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if weight is None:
            weight = torch.ones_like(y)
        if weight.ndim == 1:
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
        self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None = None
    ) -> None:
        """
        A <- A + x*x.t
        b <- b + r*x
        """
        # this also appends a column of ones to `x`
        x, y, weight = self._validate_train_inputs(x, y, weight)

        delta_A = torch.matmul(x.t(), x * weight)
        delta_A = (delta_A + delta_A.t()) / 2  # symmetrize to avoid numerical errors
        delta_b = torch.matmul(x.t(), y * weight).squeeze(-1)
        delta_sum_weight = weight.sum()

        if self.distribution_enabled:
            torch.distributed.all_reduce(delta_A)
            torch.distributed.all_reduce(delta_b)
            torch.distributed.all_reduce(delta_sum_weight)

        self._A += delta_A.to(self._A.device)
        self._b += delta_b.to(self._b.device)
        self._sum_weight += delta_sum_weight.to(self._sum_weight.device)

        self.calculate_coefs()  # update coefs after updating A and b

    def apply_discounting(self) -> None:
        """
        Apply gamma (discountting multiplier) to A and b.
        Gamma is <=1, so it reduces the effect of old data points and enabled the model to
            "forget" old data and adjust to new data distribution in non-stationary environment

        A <- A * gamma
        b <- b * gamma
        """
        if self.gamma < 1:
            logger.info(f"Applying discounting at sum_weight={self._sum_weight}")
            self._A *= self.gamma
            self._b *= self.gamma
        # don't dicount sum_weight because it's used to determine when to apply discounting

        self.calculate_coefs()  # update coefs using new A and b

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
        self._inv_A = self.matrix_inv_fallback_pinv(self.A)
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
        return f"LinearRegression(A:\n{self.A}\nb:\n{self._b})"

    from typing import List

    def compare(self, other: MuSigmaCBModel) -> str:
        """
        Compares two LinearRegression instances for equality,
        checking attributes and buffers.

        Args:
        other: The other LinearRegression instance to compare with.

        Returns:
        str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, LinearRegression):
            differences.append("other is not an instance of LinearRegression")
        assert isinstance(other, LinearRegression)
        if self.gamma != other.gamma:
            differences.append(f"gamma is different: {self.gamma} vs {other.gamma}")
        if self.l2_reg_lambda != other.l2_reg_lambda:
            differences.append(
                f"l2_reg_lambda is different: {self.l2_reg_lambda} vs {other.l2_reg_lambda}"
            )
        if self.force_pinv != other.force_pinv:
            differences.append(
                f"force_pinv is different: {self.force_pinv} vs {other.force_pinv}"
            )
        if self.distribution_enabled != other.distribution_enabled:
            differences.append(
                f"distribution_enabled is different: {self.distribution_enabled} "
                + f"vs {other.distribution_enabled}"
            )
        if not torch.allclose(self._A, other._A):
            differences.append(f"_A is different: {self._A} vs {other._A}")
        if not torch.allclose(self._b, other._b):
            differences.append(f"_b is different: {self._b} vs {other._b}")
        if not torch.allclose(self._sum_weight, other._sum_weight):
            differences.append(
                f"_sum_weight is different: {self._sum_weight} vs {other._sum_weight}"
            )
        if not torch.allclose(self._inv_A, other._inv_A):
            differences.append(f"_inv_A is different: {self._inv_A} vs {other._inv_A}")
        if not torch.allclose(self._coefs, other._coefs):
            differences.append(f"_coefs is different: {self._coefs} vs {other._coefs}")

        return "\n".join(differences)  # Join the differences with newlines
