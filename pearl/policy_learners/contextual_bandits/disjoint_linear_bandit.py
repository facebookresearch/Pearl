# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict, List

import torch

from pearl.api.action import Action
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.common.utils import ensemble_forward
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.action_utils import (
    concatenate_actions_to_state,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import nn


class DisjointLinearBandit(ContextualBanditBase):
    """
    LinearBandit for discrete action space with each action has its own linear
    regression.
    DisjointLinearBandit will be deprecated. Use DisjointBanditContainer instead.
    """

    def __init__(
        self,
        feature_dim: int,
        action_space: DiscreteActionSpace,
        exploration_module: ExplorationModule,
        l2_reg_lambda: float = 1.0,
        training_rounds: int = 100,
        batch_size: int = 128,
        state_features_only: bool = False,
    ) -> None:
        super(DisjointLinearBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        # Currently our disjoint LinUCB usecase only use LinearRegression

        # Keep list attribute since ensemble_forward requires List[nn.Module]
        self._linear_regressions_list: List[nn.Module] = [
            LinearRegression(feature_dim=feature_dim, l2_reg_lambda=l2_reg_lambda)
            for _ in range(action_space.n)
        ]
        # create nn.ModuleList so self.to(device) will move modules along
        self._linear_regressions = nn.ModuleList(self._linear_regressions_list)
        self._discrete_action_space = action_space
        self._state_features_only = state_features_only

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Assumption of input is that action in
        batch is action idx instead of action value
        Only discrete action problem will use DisjointLinearBandit
        """
        for action_idx, linear_regression in enumerate(self._linear_regressions):
            index = torch.nonzero(batch.action == action_idx, as_tuple=True)[0]
            if index.numel() == 0:
                continue
            state = torch.index_select(
                batch.state,
                dim=0,
                index=index,
            )
            if self._state_features_only:
                context = state
            else:
                # cat state with corresponding action tensor
                expanded_action = (
                    torch.Tensor(self._discrete_action_space[action_idx])
                    .unsqueeze(0)
                    .expand(state.shape[0], -1)
                    .to(batch.device)
                )
                context = torch.cat([state, expanded_action], dim=1)
            reward = torch.index_select(
                batch.reward,
                dim=0,
                index=index,
            )
            if batch.weight is not None:
                weight = torch.index_select(
                    batch.weight,
                    dim=0,
                    index=index,
                )
            else:
                weight = torch.ones(reward.shape, device=batch.device)
            linear_regression.learn_batch(
                x=context,
                y=reward,
                weight=weight,
            )

        return {}

    # pyre-fixme[14]: `act` overrides method defined in `ContextualBanditBase`
    #  inconsistently.
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        exploit: bool = False,
    ) -> Action:
        # TODO static discrete action space only
        feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=action_space,
            state_features_only=self._state_features_only,
            action_representation_module=self._action_representation_module,
        )
        # (batch_size, action_count, feature_size)

        values = ensemble_forward(
            self._linear_regressions_list, feature, use_for_loop=True
        )
        return self._exploration_module.act(
            subjective_state=feature,
            action_space=action_space,
            values=values,
            representation=self._linear_regressions_list,  # pyre-fixme[6]: unexpected type
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement when necessary")
