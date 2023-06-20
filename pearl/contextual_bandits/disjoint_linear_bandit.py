#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Dict

import torch

from pearl.api.action import Action
from pearl.contextual_bandits.contextual_bandit_base import ContextualBanditBase
from pearl.contextual_bandits.linear_regression import LinearRegression
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffer.transition import TransitionBatch

from pearl.utils.action_spaces import DiscreteActionSpace
from torch.func import stack_module_state


class DisjointLinearBandit(ContextualBanditBase):
    """
    LinearBandit for discrete action space with each action has its own linear regression
    """

    def __init__(
        self,
        feature_dim: int,
        action_count: int,
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(DisjointLinearBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        # Currently our disjoint LinUCB usecase only use LinearRegression
        self._linear_regressions = [
            LinearRegression(feature_dim=feature_dim) for _ in range(action_count)
        ]

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Assumption of input is that action in batch is action idx instead of action value
        Only discrete action problem will use DisjointLinearBandit
        """

        for action_idx, linear_regression in enumerate(self._linear_regressions):
            index = torch.nonzero(batch.action == action_idx).squeeze()
            if index.numel() == 0:
                continue
            context = torch.index_select(
                batch.state,
                dim=0,
                index=index,
            )
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
                weight = torch.ones(reward.shape)
            linear_regression.train(
                x=context,
                y=reward,
                weight=weight,
            )

        return {}

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        _exploit: bool = False,
    ) -> Action:
        # TODO static discrete action space only without action vector
        assert action_space.action_dim == 0
        subjective_state = subjective_state.view(
            -1, self._feature_dim
        )  # reshape to (batch_size, feature_dim)
        # followed example in https://pytorch.org/docs for ensembling
        def wrapper(params, buffers):
            return torch.func.functional_call(
                self._linear_regressions[0], (params, buffers), subjective_state
            )

        params, buffers = stack_module_state(self._linear_regressions[: action_space.n])
        values = torch.vmap(wrapper, (0, 0))(
            params, buffers
        )  # (action_count, batch_size)
        # change shape to (batch_size, action_count)
        values = values.permute(1, 0)
        return self._exploration_module.act(
            subjective_state=subjective_state,
            action_space=action_space,
            values=values,
            representation=self._linear_regressions,
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement when necessary")
