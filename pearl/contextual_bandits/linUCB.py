#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Any, Dict

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.contextual_bandits.contextual_bandit_base import ContextualBanditBase
from pearl.contextual_bandits.linear_regression import AvgWeightLinearRegression
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffer.transition import TransitionBatch


class LinUCB(ContextualBanditBase):
    """
    Linear UCB Policy Learner
    paper: https://arxiv.org/pdf/1003.0146.pdf

    If people use the disjoint mode in above papar, please check class DisjointLinUCB
    """

    def __init__(
        self,
        feature_dim: int,
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(LinUCB, self).__init__(
            state_dim=feature_dim,  # not useful yet
            action_space=None,  # not useful yet
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
        action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        raise NotImplementedError("Integrate with UCB exploration module")
