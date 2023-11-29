from abc import abstractmethod
from typing import Type

import torch
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.common.value_networks import QuantileQValueNetwork
from pearl.neural_networks.sequential_decision_making.q_value_network import (
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

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, **options) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self._action_space = None

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def filter_action(self, subjective_state: SubjectiveState) -> ActionSpace:
        return self._action_space

    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        pass

    def learn_batch(self, batch: TransitionBatch) -> None:
        pass

    # risk sentitive safe rl methods use this to compute q values from a q value distribution.
    @abstractmethod
    def get_q_values_under_risk_metric(
        self,
        batch: TransitionBatch,
        q_value_distribution: Type[DistributionalQValueNetwork],
    ) -> torch.Tensor:
        pass


class RiskNeutralSafetyModule(RiskSensitiveSafetyModule):
    """
    A safety module that computes q values as expectation of a q value distribution.
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, **options) -> None:
        super(RiskNeutralSafetyModule, self).__init__()

    # pyre-fixme[3]: Return type must be annotated.
    def __str__(self):
        return f"Safety module type {self.__class__.__name__}"

    # pyre-fixme[14]: `get_q_values_under_risk_metric` overrides method defined in
    #  `RiskSensitiveSafetyModule` inconsistently.
    def get_q_values_under_risk_metric(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        q_value_distribution_network: Type[DistributionalQValueNetwork],
    ) -> torch.Tensor:

        """Returns Q(s, a), given s and a
        Args:
            state_batch: a batch of state tensors (batch_size, state_dim)
            action_batch: a batch of action tensors (batch_size, action_dim)
            q_value_distribution_network: a distributional q value network that approximates the return distribution
        Returns:
            Q-values of (state, action) pairs: (batch_size) under a risk neutral measure, that is, Q(s, a) = E[Z(s, a)]
        """
        # pyre-fixme[20]: Argument `action_batch` expected.
        q_value_distribution = q_value_distribution_network.get_q_value_distribution(
            state_batch,
            action_batch,
        )

        return q_value_distribution.mean(dim=-1)


class QuantileNetworkMeanVarianceSafetyModule(RiskSensitiveSafetyModule):
    """
    A safety module that computes q values as a weighted linear combination of mean and variance of the q value distribution.
    Q(s, a) = E[Z(s, a)] - (beta) * Var[Z(s, a)]
    """

    def __init__(
        self,
        variance_weighting_coefficient: float,
    ) -> None:
        super(QuantileNetworkMeanVarianceSafetyModule, self).__init__()
        self._beta = variance_weighting_coefficient

    # pyre-fixme[3]: Return type must be annotated.
    def __str__(self):
        return f"Safety module type {self.__class__.__name__}"

    # pyre-fixme[14]: `get_q_values_under_risk_metric` overrides method defined in
    #  `RiskSensitiveSafetyModule` inconsistently.
    def get_q_values_under_risk_metric(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        q_value_distribution_network: Type[
            DistributionalQValueNetwork
        ] = QuantileQValueNetwork,
    ) -> torch.Tensor:
        # pyre-fixme[20]: Argument `action_batch` expected.
        q_value_distribution = q_value_distribution_network.get_q_value_distribution(
            state_batch,
            action_batch,
        )
        """
        variance computation:
            - sum_{i=0}^{N-1} (tau_{i+1} - tau_{i}) * (q_value_distribution_{tau_i} - mean_value)^2
        """
        mean_value = q_value_distribution.mean(dim=-1, keepdim=True)
        quantiles = q_value_distribution_network.quantiles
        quantile_differences = quantiles[1:] - quantiles[:-1]
        variance = (
            quantile_differences * torch.square(q_value_distribution - mean_value)
        ).sum(dim=-1, keepdim=True)
        variance_adjusted_mean = (mean_value - (self._beta * variance)).view(-1)
        return variance_adjusted_mean
