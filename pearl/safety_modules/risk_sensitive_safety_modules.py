# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import abstractmethod
from typing import List

import torch
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    DistributionalQValueNetwork,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.safety_module import SafetyModule
from torch import Tensor


class RiskSensitiveSafetyModule(SafetyModule):
    """
    A safety module that computes q values from a q value distribution given a risk measure.
    Base class for different risk metrics, e.g. mean-variance, Value-at-risk (VaR) etc.
    """

    def filter_action(
        self, subjective_state: SubjectiveState, action_space: ActionSpace
    ) -> ActionSpace:
        return action_space

    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        pass

    def learn_batch(self, batch: TransitionBatch) -> None:
        pass

    # risk sentitive safe rl methods use this to compute q values from a q value distribution.
    @abstractmethod
    def get_q_values_under_risk_metric(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        q_value_distribution_network: DistributionalQValueNetwork,
    ) -> torch.Tensor:
        pass


class RiskNeutralSafetyModule(RiskSensitiveSafetyModule):
    """
    A safety module that computes q values as expectation of a q value distribution.
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return f"Safety module type {self.__class__.__name__}"

    def get_q_values_under_risk_metric(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        q_value_distribution_network: DistributionalQValueNetwork,
    ) -> torch.Tensor:
        """Returns Q(s, a), given s and a
        Args:
            state_batch: a batch of state tensors (batch_size, state_dim)
            action_batch: a batch of action tensors
            (batch_size, number_query_actions, action_dim) or (batch_size, action_dim)
            q_value_distribution_network: a distributional q value network that
                                          approximates the return distribution
        Returns:
            Q-values of (state, action) pairs: (batch_size) under a risk neutral measure,
            that is, Q(s, a) = E[Z(s, a)]
        """
        assert len(state_batch.shape) == 2
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        q_value_distribution = q_value_distribution_network.get_q_value_distribution(
            state_batch, action_batch
        )

        return q_value_distribution.mean(dim=-1)

    def compare(self, other: SafetyModule) -> str:
        """
        Compares two RiskNeutralSafetyModule instances for equality.

        Since this module has no attributes or buffers to compare,
        it only checks if the `other` object is an instance of the same class.

        Args:
          other: The other SafetyModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, RiskNeutralSafetyModule):
            differences.append("other is not an instance of RiskNeutralSafetyModule")

        return "\n".join(differences)


class QuantileNetworkMeanVarianceSafetyModule(RiskSensitiveSafetyModule):
    """
    A safety module that computes q values as a weighted linear combination
    of mean and variance of the q value distribution.
    Q(s, a) = E[Z(s, a)] - (beta) * Var[Z(s, a)]
    """

    def __init__(
        self,
        variance_weighting_coefficient: float,
    ) -> None:
        super().__init__()
        self._beta = variance_weighting_coefficient

    def __str__(self) -> str:
        return f"Safety module type {self.__class__.__name__}"

    def get_q_values_under_risk_metric(
        self,
        state_batch: Tensor,  # (batch_size, state_dim)
        action_batch: Tensor,  # (batch_size, number_query_actions, state_dim)
        q_value_distribution_network: DistributionalQValueNetwork,
    ) -> torch.Tensor:
        q_value_distribution = q_value_distribution_network.get_q_value_distribution(
            state_batch,
            action_batch,
        )  # (batch_size, number_query_actions, num quantiles)
        """
        variance computation:
            - sum_{i=0}^{N-1} (tau_{i+1} - tau_{i}) * (q_value_distribution_{tau_i} - mean_value)^2
        """
        assert len(state_batch.shape) == 2
        assert len(action_batch.shape) == 3
        mean_value = q_value_distribution.mean(
            dim=-1, keepdim=True
        )  # (batch_size, number_query_actions, 1)
        quantiles = q_value_distribution_network.quantiles  # (num quantiles + 1,)
        quantile_differences = quantiles[1:] - quantiles[:-1]  # (num quantiles)
        variance = (
            quantile_differences * torch.square(q_value_distribution - mean_value)
        ).sum(dim=-1, keepdim=True)  # (batch_size, number_query_actions, 1)
        variance_adjusted_mean = mean_value - (
            self._beta * variance
        )  # (batch_size, number_query_actions, 1)
        return variance_adjusted_mean.squeeze(-1)

    def compare(self, other: SafetyModule) -> str:
        """
        Compares two QuantileNetworkMeanVarianceSafetyModule instances for equality,
        checking the variance_weighting_coefficient.

        Args:
          other: The other SafetyModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, QuantileNetworkMeanVarianceSafetyModule):
            differences.append(
                "other is not an instance of QuantileNetworkMeanVarianceSafetyModule"
            )
        else:
            if self._beta != other._beta:
                differences.append(f"_beta is different: {self._beta} vs {other._beta}")

        return "\n".join(differences)
