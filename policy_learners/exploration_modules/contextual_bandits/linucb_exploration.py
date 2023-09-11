import logging
from typing import Any

import torch

from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    UCBExplorationBase,
)
from pearl.utils.functional_utils.learning.linear_regression import (
    AvgWeightLinearRegression,
)
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


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


def calculate_variance(
    subjective_state: torch.Tensor, representation: torch.nn.Module
) -> torch.Tensor:
    A_inv = representation.inv_A
    sum_weight = (
        representation.sum_weight
        if isinstance(representation, AvgWeightLinearRegression)
        else 1
    )
    uncertainty = torch.sqrt(batch_quadratic_form(subjective_state, A_inv) / sum_weight)
    return uncertainty


class LinUCBExploration(UCBExplorationBase):
    """
    Exploration module for linear UCB with disjoint linear models
    paper: https://arxiv.org/pdf/1003.0146.pdf
    """

    def uncertainty(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        representation: Any = None,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state is the feature vector, if action feature and state feature needs to be concat
                it should have been done at caller side, shape(batch_size, action_count, feature_dim) or (batch_size, feature_dim)
            representation is one linear regression
        Returns:
            uncertainty with shape (batch_size, action_count) or (batch_size, 1)
        """
        uncertainty = calculate_variance(subjective_state, representation)
        nan_check = torch.isnan(uncertainty)
        if torch.any(nan_check):
            # nan doesnt make sense, it's usually caused by bad training data
            # print out warning and set to 0
            logging.warning("nan appeared in ucb uncertainty")
            uncertainty[nan_check] = 0
        return uncertainty


class DisjointLinUCBExploration(LinUCBExploration):
    """
    Same as LinUCBExploration, just that now different action has different linear regression
    """

    def uncertainty(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        representation: Any = None,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state: this is feature vector in shape, batch_size, action_count, feature
            representation: unlike LinUCBExploration, here it is a list for different actions
        """
        uncertainty = []
        for i, linear_regression in enumerate(representation):
            uncertainty.append(
                super(DisjointLinUCBExploration, self).uncertainty(
                    subjective_state=subjective_state[
                        :, i, :
                    ],  # different action has different feature
                    available_action_space=available_action_space,
                    representation=linear_regression,
                )
            )
        uncertainty = torch.stack(uncertainty)
        # change from shape(action_count, batch_size) to shape(batch_size, action_count)
        uncertainty = uncertainty.permute(1, 0)
        return uncertainty
