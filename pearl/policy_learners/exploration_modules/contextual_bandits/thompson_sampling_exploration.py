# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, List

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.policy_learners.exploration_modules import ExplorationModule
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


# TODO: generalize for non-linear models
class ThompsonSamplingExplorationLinear(ScoreExplorationBase):
    """
    Thompson Sampling exploration module for the joint linear bandits.
    """

    def __init__(
        self,
        enable_efficient_sampling: bool = False,
    ) -> None:
        super().__init__()
        self._enable_efficient_sampling = enable_efficient_sampling

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        representation: torch.nn.Module | None = None,
        exploit_action: Action | None = None,
    ) -> torch.Tensor:
        """
        Given the linear bandit model, sample its parameters,
        and multiplies with feature to get predicted score.
        """
        assert isinstance(action_space, DiscreteActionSpace)
        assert representation is not None
        if self._enable_efficient_sampling:
            expected_reward = representation(subjective_state)
            # batch_size, action_count, 1
            assert expected_reward.shape == subjective_state.shape[:-1]
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            sigma = representation.calculate_sigma(subjective_state)
            # batch_size, action_count, 1
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

        return scores.view(-1, action_space.n)

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two ThompsonSamplingExplorationLinear instances for equality,
        checking attributes.

        Args:
          other: The other ExplorationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, ThompsonSamplingExplorationLinear):
            differences.append(
                "other is not an instance of ThompsonSamplingExplorationLinear"
            )
        else:
            if self._enable_efficient_sampling != other._enable_efficient_sampling:
                differences.append(
                    f"_enable_efficient_sampling is different: {self._enable_efficient_sampling} "
                    + "vs {other._enable_efficient_sampling}"
                )

        return "\n".join(differences)


class ThompsonSamplingExplorationLinearDisjoint(ThompsonSamplingExplorationLinear):
    """
    Thompson Sampling exploration module for the disjoint linear bandits.
    """

    def __init__(
        self,
        enable_efficient_sampling: bool = False,
    ) -> None:
        super().__init__(enable_efficient_sampling=enable_efficient_sampling)

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        representation: Any = None,
        exploit_action: Action | None = None,
    ) -> torch.Tensor:
        assert isinstance(action_space, DiscreteActionSpace)
        # DisJoint Linear Bandits
        # The representation is a list for different actions.
        scores = []
        for i, model in enumerate(representation):
            # subjective_state is in shape of batch_size, action_count, feature_dim
            single_action_space = DiscreteActionSpace([action_space[i]])
            score = super().get_scores(  # call get_scores() from joint TS
                subjective_state=subjective_state[:, i, :],
                action_space=single_action_space,
                values=values,
                representation=model,
                exploit_action=exploit_action,
            )
            scores.append(score)
        scores = torch.stack(scores)
        return scores.view(-1, action_space.n)

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two ThompsonSamplingExplorationLinearDisjoint instances for equality.

        Args:
          other: The other ExplorationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, ThompsonSamplingExplorationLinearDisjoint):
            differences.append(
                "other is not an instance of ThompsonSamplingExplorationLinearDisjoint"
            )
        else:
            if self._enable_efficient_sampling != other._enable_efficient_sampling:
                differences.append(
                    f"_enable_efficient_sampling is different: {self._enable_efficient_sampling} "
                    + f"vs {other._enable_efficient_sampling}"
                )

        return "\n".join(differences)
