from typing import Any

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.common.value_exploration_base import (
    ValueExplorationBase,
)
from pearl.utils.functional_utils.learning.linear_regression import LinearRegression
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class ThompsonSamplingExplorationLinear(ValueExplorationBase):
    """
    Thompson Sampling exploration module for the joint linear bandits.
    """

    def __init__(
        self,
        enable_efficient_sampling: bool = False,
    ) -> None:
        super(ThompsonSamplingExplorationLinear, self).__init__()
        self._enable_efficient_sampling = enable_efficient_sampling

    def sampling(
        self, subjective_state: SubjectiveState, linear_bandit_model: torch.nn.Module
    ):
        """
        Given the linear bandit model, sample its parameters, and multiplies with feature to get predicted score.
        """
        if self._enable_efficient_sampling:
            expected_reward = linear_bandit_model(
                subjective_state
            )  # batch_size, action_count, 1
            assert expected_reward.shape == subjective_state.shape[:-1]
            sigma = linear_bandit_model.calculate_sigma(
                subjective_state
            )  # batch_size, action_count, 1
            assert sigma.shape == subjective_state.shape[:-1]
            score = torch.normal(mean=expected_reward, std=sigma)
        else:
            thompson_sampling_coefs = (
                torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=linear_bandit_model.coefs,
                    precision_matrix=linear_bandit_model.A,
                ).sample()
            )
            score = torch.matmul(
                LinearRegression.append_ones(subjective_state),
                thompson_sampling_coefs.t(),
            )
        return score

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        values: torch.Tensor,
        representation: Any = None,
        exploit_action: Action = None,
    ) -> Action:

        # DisJoint Linear Bandits
        # The representation is a nn.Module, e.g., a linear regression model.
        # subjective_state is in shape of (batch_size, feature_dim)
        score = self.sampling(subjective_state, representation)

        # get the best action with the highest value
        values_sampled = score.view(-1, action_space.n)  # (batch_size, )
        selected_action = torch.argmax(values_sampled, dim=1).squeeze()
        return selected_action


class ThompsonSamplingExplorationLinearDisjoint(ThompsonSamplingExplorationLinear):
    """
    Thompson Sampling exploration module for the disjoint linear bandits.
    """

    def __init__(
        self,
    ) -> None:
        super(ThompsonSamplingExplorationLinearDisjoint, self).__init__()

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        values: torch.Tensor,
        representation: Any = None,
        exploit_action: Action = None,
    ) -> Action:

        # DisJoint Linear Bandits
        # The representation is a list for different actions.
        values_sampled = []
        for i, model in enumerate(representation):
            # subjective_state is in shape of batch_size, action_count, feature_dim
            score = self.sampling(subjective_state[:, i, :], model)
            values_sampled.append(score)
        values_sampled = torch.stack(values_sampled)

        # get the best action with the highest value
        values_sampled = values_sampled.view(
            -1, action_space.n
        )  # batch_size, action_count
        selected_action = torch.argmax(values_sampled, dim=1).squeeze()
        return selected_action
