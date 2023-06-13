#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Any, Dict

import torch
from pearl.api.action import Action
from pearl.contextual_bandits.contextual_bandit_base import ContextualBanditBase
from pearl.contextual_bandits.linear_regression import (
    AvgWeightLinearRegression,
    LinearRegression,
)
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.exploration_module.ucb_exploration import UCBExplorationBase
from pearl.replay_buffer.transition import TransitionBatch
from pearl.utils.action_spaces import DiscreteActionSpace


class LinearBandit(ContextualBanditBase):
    """
    Policy Learner for Contextual Bandit with Linear Policy
    """

    def __init__(
        self,
        feature_dim: int,
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(LinearBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        self._linear_regression = AvgWeightLinearRegression(feature_dim=feature_dim)

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        A <- A + x*x.t
        b <- b + r*x
        """
        self._linear_regression.train(
            x=torch.cat([batch.state, batch.action], dim=1),
            y=batch.reward,
            weight=batch.weight,
        )
        return {}

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        exploit: bool = False,
    ) -> Action:
        """
        Args:
            subjective_state - state will be applied to different action vectors in action_space
            action_space contains a list of action vector, currenly only support static space
        Return:
            action index chosen given state and action vectors
        """
        action_dim = action_space.action_dim
        action_count = action_space.n

        subjective_state = subjective_state.view(
            -1, self._feature_dim - action_dim
        )  # reshape to (batch_size, state_dim)
        batch_size = subjective_state.shape[0]

        expanded_state = subjective_state.unsqueeze(1).repeat(
            1, action_count, 1
        )  # expand to (batch_size, action_count, state_dim)

        actions = action_space.to_tensor()  # (action_count, action_dim)
        expanded_action = actions.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # batch_size, action_count, action_dim
        new_feature = torch.cat(
            [expanded_state, expanded_action], dim=2
        )  # batch_size, action_count, feature_dim
        values = self._linear_regression(new_feature)  # (batch_size, action_count)
        assert values.shape == (batch_size, action_count)
        return self._exploration_module.act(
            subjective_state=subjective_state,
            action_space=action_space,
            values=values,
            representation=self._linear_regression,
        )

    @staticmethod
    def get_linucb_scores(
        subjective_state: SubjectiveState,
        feature_dim: int,
        exploration_module: UCBExplorationBase,
        linear_regression: LinearRegression,
    ) -> torch.Tensor:
        # currently we only support joint ucb with LinearBandit
        # which means we call get_scores N times for N actions
        # for disjoint ucb, please use DisjointLinearBandit
        available_action_space = DiscreteActionSpace([0])
        subjective_state = subjective_state.view(
            -1, feature_dim
        )  # reshape to (batch_size, feature_dim)
        values = linear_regression(subjective_state)  # (batch_size, )
        values = values.unsqueeze(dim=1)  # change to (batch_size, 1)
        # get_scores returns (batch_size, action_count) or (action_count)
        # here our action count is 1, so delete that dimension by squeeze
        return exploration_module.get_ucb_scores(
            subjective_state=subjective_state,
            values=values,
            available_action_space=available_action_space,
            # for linear bandit, all actions share same linear regression
            representation={
                action: linear_regression for action in range(available_action_space.n)
            },
        ).squeeze()

    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ) -> torch.Tensor:
        """
        Returns:
            UCB scores when exploration module is UCB
            Shape is (batch)
        """
        # TODO generalize for all kinds of exploration module
        assert isinstance(self._exploration_module, UCBExplorationBase)
        return LinearBandit.get_linucb_scores(
            subjective_state=subjective_state,
            feature_dim=self._feature_dim,
            exploration_module=self._exploration_module,
            linear_regression=self._linear_regression,
        )
