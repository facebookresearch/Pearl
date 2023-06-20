from typing import Any

import torch

from pearl.api.state import SubjectiveState
from pearl.contextual_bandits.linear_regression import AvgWeightLinearRegression
from pearl.policy_learners.exploration_module.ucb_exploration import UCBExplorationBase
from pearl.utils.action_spaces import DiscreteActionSpace


class LinUCBExploration(UCBExplorationBase):
    """
    Exploration module for linear UCB with disjoint linear models
    paper: https://arxiv.org/pdf/1003.0146.pdf
    """

    def batch_quadratic_form(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the quadratic form x^T * A * x for a batched input x.
        The calcuation of pred_sigma (uncertainty) in LinUCB is done by quadratic form x^T * A^{-1} * x.
        Inspired by https://stackoverflow.com/questions/18541851/calculate-vt-a-v-for-a-matrix-of-vectors-v
        This is a vectorized implementation of out[i] = x[i].t() @ A @ x[i]
        x shape: (Batch, Feature_dim)
        A shape: (Feature_dim, Feature_dim)
        output shape: (Batch)
        """
        x = x.view(-1, A.shape[0])  # (batch_size, feature_size)
        return (torch.matmul(x, A) * x).sum(-1)

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
        A_inv = representation.inv_A
        sum_weight = (
            representation.sum_weight
            if isinstance(representation, AvgWeightLinearRegression)
            else 1
        )
        uncertainty = torch.sqrt(
            self.batch_quadratic_form(subjective_state, A_inv) / sum_weight
        )
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
            representation: unlike LinUCBExploration, here it is a list for different actions
        """
        uncertainty = []
        for linear_regression in representation:
            uncertainty.append(
                super(DisjointLinUCBExploration, self).uncertainty(
                    subjective_state=subjective_state,
                    available_action_space=available_action_space,
                    representation=linear_regression,
                )
            )
        uncertainty = torch.stack(uncertainty)
        # change from shape(action_count, batch_size) to shape(batch_size, action_count)
        uncertainty = uncertainty.permute(1, 0)
        return uncertainty
