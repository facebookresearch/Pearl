from typing import List, Optional

import torch

from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.contextual_bandits.ucb_exploration import (
    UCBExploration,
)
from pearl.utils.tensor_like import assert_is_tensor_like


class LinUCBExploration(UCBExploration):
    """
    Exploration module for linear UCB with joint linear models
    paper: https://arxiv.org/pdf/1003.0146.pdf
    """

    def sigma(
        self,
        subjective_state: SubjectiveState,
        representation: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state: feature vector (either state,
            or state and action features after concatenation)
            Shape should be either (batch_size, action_count, feature_dim) or
            (batch_size, feature_dim).
        Returns:
            sigma with shape (batch_size, action_count) or (batch_size, 1)
        """
        assert representation is not None
        sigma = representation.calculate_sigma(subjective_state)
        nan_check = torch.isnan(sigma)
        sigma = torch.where(nan_check, torch.zeros_like(sigma), sigma)
        return sigma


class DisjointLinUCBExploration(LinUCBExploration):
    """
    Same as LinUCBExploration, just that now different action has different linear regression
    """

    # pyre-fixme[14]: `sigma` overrides method defined in `LinUCBExploration`
    #  inconsistently.
    def sigma(
        self,
        subjective_state: SubjectiveState,
        representation: Optional[List[torch.nn.Module]] = None,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state: this is feature vector in shape, batch_size, action_count, feature
            representation: unlike LinUCBExploration, here it is a list for different actions
        """
        assert representation is not None
        subjective_state = assert_is_tensor_like(subjective_state)
        sigma = []
        for i, linear_regression in enumerate(representation):
            sigma.append(
                super(DisjointLinUCBExploration, self).sigma(
                    subjective_state=subjective_state[
                        :, i, :
                    ],  # different action has different feature
                    representation=linear_regression,
                )
            )
        sigma = torch.stack(sigma)
        # change from shape(action_count, batch_size) to shape(batch_size, action_count)
        sigma = sigma.permute(1, 0)
        return sigma
