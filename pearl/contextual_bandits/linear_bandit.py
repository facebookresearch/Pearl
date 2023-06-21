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
from pearl.contextual_bandits.linucb_exploration import LinUCBExploration
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
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
        # It doesnt make sense to call act if we are not working with action vector
        assert action_space.action_dim > 0
        action_count = action_space.n
        new_feature = action_space.cat_state_tensor(subjective_state)
        values = self._linear_regression(new_feature)  # (batch_size, action_count)
        assert values.shape == (new_feature.shape[0], action_count)
        return self._exploration_module.act(
            # TODO we might want to name this new_feature
            # so exploration module doesnt need to worry about broadcast state to different action vector
            subjective_state=new_feature,
            action_space=action_space,
            values=values,
            representation=self._linear_regression,
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace = None,
    ) -> torch.Tensor:
        """
        Returns:
            UCB scores when exploration module is UCB
            Shape is (batch)
        """
        # TODO generalize for all kinds of exploration module
        assert isinstance(self._exploration_module, LinUCBExploration)
        feature = (
            action_space.cat_state_tensor(subjective_state)
            if action_space is not None
            else subjective_state
        )
        return self._exploration_module.get_ucb_scores(
            subjective_state=feature,
            values=self._linear_regression(feature),
            # when action_space is None, we are querying score for one action
            available_action_space=action_space
            if action_space is not None
            else DiscreteActionSpace([0]),
            representation=self._linear_regression,
        ).squeeze()
