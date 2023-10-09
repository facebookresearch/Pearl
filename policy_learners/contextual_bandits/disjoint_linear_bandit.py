#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Dict

from warnings import warn

import torch

from pearl.api.action import Action
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.common.utils import ensemble_forward
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.device import get_pearl_device
from pearl.utils.functional_utils.learning.linear_regression import LinearRegression
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class DisjointLinearBandit(ContextualBanditBase):
    """
    LinearBandit for discrete action space with each action has its own linear regression
    """

    def __init__(
        self,
        feature_dim: int,
        action_space: DiscreteActionSpace,
        exploration_module: ExplorationModule,
        l2_reg_lambda: float = 1.0,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        warn(
            "DisjointLinearBandit will be deprecated. Use DisjointBanditContainer instead",
            DeprecationWarning,
        )
        super(DisjointLinearBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.device = get_pearl_device()
        # Currently our disjoint LinUCB usecase only use LinearRegression
        # pyre-fixme[4]: Attribute must be annotated.
        self._linear_regressions = torch.nn.ModuleList(
            [
                LinearRegression(feature_dim=feature_dim, l2_reg_lambda=l2_reg_lambda)
                for _ in range(action_space.n)
            ]
        ).to(self.device)
        self._discrete_action_space = action_space

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Assumption of input is that action in batch is action idx instead of action value
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
            # cat state with corresponding action tensor
            expanded_action = (
                torch.Tensor(self._discrete_action_space[action_idx])
                .unsqueeze(0)
                .expand(state.shape[0], -1)
                .to(self.device)
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
                weight = torch.ones(reward.shape, device=self.device)
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
        _exploit: bool = False,
    ) -> Action:
        # TODO static discrete action space only, so here action_space should == self._discrete_action_space
        feature = self._discrete_action_space.cat_state_tensor(
            subjective_state=subjective_state
        )  # batch_size, action_count, feature_size

        values = ensemble_forward(self._linear_regressions, feature)

        return self._exploration_module.act(
            subjective_state=feature,
            action_space=action_space,
            values=values,
            representation=self._linear_regressions,
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement when necessary")
