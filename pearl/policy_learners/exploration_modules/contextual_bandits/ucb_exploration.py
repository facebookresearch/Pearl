# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


# TODO: Assumes discrete gym action space
class UCBExploration(ScoreExplorationBase):
    """
    UCB exploration module.
    """

    def __init__(self, alpha: float) -> None:
        super(UCBExploration, self).__init__()
        self._alpha = alpha

    def sigma(
        self,
        subjective_state: SubjectiveState,
        representation: torch.nn.Module,
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
        sigma = representation.calculate_sigma(subjective_state)
        nan_check = torch.isnan(sigma)
        sigma = torch.where(nan_check, torch.zeros_like(sigma), sigma)
        return sigma

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        values: torch.Tensor,
        action_space: ActionSpace,
        representation: Optional[torch.nn.Module] = None,
        exploit_action: Optional[Action] = None,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state is in shape of (batch_size, feature_size)
            values is in shape of (batch_size, action_count)
        Returns:
            return shape(batch_size, action_count)
        or
        Args:
            subjective_state is in shape of (feature_size)
            values is in shape of (action_count)
        Returns:
            return shape(action_count)
        """
        assert isinstance(action_space, DiscreteActionSpace)
        action_count = action_space.n
        values = values.view(-1, action_count)  # (batch_size, action_count)
        sigma = self.sigma(
            subjective_state=subjective_state,
            # pyre-fixme[6]: For 2nd argument expected `Module` but got
            #  `Optional[Module]`.
            representation=representation,
        )
        # a safe check before reshape sigma into values
        sigma = sigma.view(values.shape)
        ucb_scores = values + self._alpha * sigma
        return ucb_scores.view(-1, action_count)  # batch_size, action_count


class DisjointUCBExploration(UCBExploration):
    """
    Same as UCBExploration, but with a separate bandit model for each action
    """

    # pyre-fixme[14]: `sigma` overrides method defined in `UCBExploration`
    #  inconsistently.
    def sigma(
        self,
        subjective_state: SubjectiveState,
        representation: torch.nn.ModuleList,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state: this is feature vector in shape, batch_size, action_count, feature
            representation: a list of bandit models, one per action (arm)
        """
        sigmas = []
        for i, arm_model in enumerate(representation):
            sigmas.append(
                super(DisjointUCBExploration, self).sigma(
                    subjective_state=subjective_state[:, i, :],
                    representation=arm_model,
                )
            )
        sigmas = torch.cat(sigmas, dim=1)
        return sigmas


class VanillaUCBExploration(UCBExploration):
    """
    Vanilla UCB exploration module with counter.
    """

    def __init__(self) -> None:
        super(VanillaUCBExploration, self).__init__(alpha=1)
        # pyre-fixme[4]: Attribute must be annotated.
        self.action_execution_count = {}
        # pyre-fixme[4]: Attribute must be annotated.
        self.action_executed = torch.tensor(1)

    # pyre-fixme[14]: `sigma` overrides method defined in `UCBExploration`
    #  inconsistently.
    def sigma(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        representation: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        assert isinstance(action_space, DiscreteActionSpace)
        exploration_bonus = torch.zeros((action_space.n))  # (action_space_size)
        for action in action_space.actions:
            if action not in self.action_execution_count:
                self.action_execution_count[action] = 1
            exploration_bonus[action] = torch.sqrt(
                torch.log(self.action_executed) / self.action_execution_count[action]
            )
        return exploration_bonus

    # TODO: We should make discrete action space itself iterable
    # pyre-fixme[14]: `act` overrides method defined in `ScoreExplorationBase`
    #  inconsistently.
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        representation: Any = None,
        exploit_action: Optional[Action] = None,
    ) -> Action:
        selected_action = super().act(
            subjective_state,
            action_space,
            values,
            representation,
            exploit_action,
        )
        self.action_execution_count[selected_action] += 1
        self.action_executed += 1
        return selected_action
