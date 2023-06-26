from typing import Any

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_module.value_exploration_base import (
    ValueExplorationBase,
)

from pearl.utils.action_spaces import DiscreteActionSpace


class ThompsonSamplingExplorationLinearDisjoint(ValueExplorationBase):
    """
    Thompson Sampling exploration module for the disjoint linear bandits.
    """

    def __init__(
        self,
    ) -> None:
        super(ThompsonSamplingExplorationLinearDisjoint, self).__init__()

    def sampling(
        self, subjective_state: SubjectiveState, linear_bandit_model: torch.nn.Module
    ):
        """
        Given the linear bandit model, sample its parameters, and multiplies with feature to get predicted score.
        """
        thompson_sampling_coefs = (
            torch.distributions.multivariate_normal.MultivariateNormal(
                loc=linear_bandit_model.coefs,
                covariance_matrix=linear_bandit_model.inv_A,
            ).sample()
        )
        score = torch.matmul(subjective_state, thompson_sampling_coefs.t())
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
