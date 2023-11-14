from typing import Any, Optional

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.utils.functional_utils.learning.linear_regression import LinearRegression
from pearl.utils.instantiations.action_spaces.action_spaces import ActionSpace


# TODO: generalize for non-linear models
class ThompsonSamplingExplorationLinear(ScoreExplorationBase):
    """
    Thompson Sampling exploration module for the joint linear bandits.
    """

    def __init__(
        self,
        enable_efficient_sampling: bool = False,
    ) -> None:
        super(ThompsonSamplingExplorationLinear, self).__init__()
        self._enable_efficient_sampling = enable_efficient_sampling

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        representation: Optional[torch.nn.Module] = None,
        exploit_action: Action = None,
    ) -> torch.Tensor:
        """
        Given the linear bandit model, sample its parameters, and multiplies with feature to get predicted score.
        """
        assert representation is not None
        batch_size = subjective_state.shape[0]
        if self._enable_efficient_sampling:
            expected_reward = representation(
                subjective_state
            )  # batch_size, action_count, 1
            assert expected_reward.shape == subjective_state.shape[:-1]
            sigma = representation.calculate_sigma(
                subjective_state
            )  # batch_size, action_count, 1
            assert sigma.shape == subjective_state.shape[:-1]
            scores = torch.normal(mean=expected_reward, std=sigma)
        else:
            thompson_sampling_coefs = (
                torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=representation.coefs,
                    precision_matrix=representation.A,
                ).sample()
            )
            scores = torch.matmul(
                LinearRegression.append_ones(subjective_state),
                thompson_sampling_coefs.t(),
            )
        return scores.view(batch_size, -1)


class ThompsonSamplingExplorationLinearDisjoint(ThompsonSamplingExplorationLinear):
    """
    Thompson Sampling exploration module for the disjoint linear bandits.
    """

    def __init__(
        self,
        enable_efficient_sampling: bool = False,
    ) -> None:
        super(ThompsonSamplingExplorationLinearDisjoint, self).__init__(
            enable_efficient_sampling=enable_efficient_sampling
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        representation: Any = None,
        exploit_action: Action = None,
    ) -> torch.Tensor:

        # DisJoint Linear Bandits
        # The representation is a list for different actions.
        scores = []
        for i, model in enumerate(representation):
            # subjective_state is in shape of batch_size, action_count, feature_dim
            score = super(
                ThompsonSamplingExplorationLinearDisjoint, self
            ).get_scores(  # call get_scores() from joint TS
                subjective_state=subjective_state[:, i, :],
                action_space=action_space,
                values=values,
                representation=model,
                exploit_action=exploit_action,
            )
            scores.append(score)
        scores = torch.stack(scores)

        # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
        return scores.view(-1, action_space.n)
