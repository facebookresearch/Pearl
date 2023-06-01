#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from collections import defaultdict
from typing import Any, Dict

import torch

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.contextual_bandits.contextual_bandit_base import ContextualBanditBase
from pearl.contextual_bandits.linear_regression import LinearRegression
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffer.transition import TransitionBatch


class DisjointLinUCB(ContextualBanditBase):
    """
    Linear UCB Policy Learner
    For discrete action space only
    Only disjoint mode for now
    paper: https://arxiv.org/pdf/1003.0146.pdf
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        exploration_module: ExplorationModule,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(DisjointLinUCB, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        # Currently our disjoint LinUCB usecase only use LinearRegression
        self._linear_regressions = {
            action_idx: LinearRegression(feature_dim=state_dim)
            for action_idx in range(action_space.n)
        }

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        # group batch data by action
        # TODO There are some assumption here. Resolve in the future when we get a better idea
        # only discrete action problem will use DisjointLinUCB, so action in replay buffer is
        # action idx instead of action value as a vector
        xs = defaultdict(list)
        rewards = defaultdict(list)
        weights = defaultdict(list)
        for i, action in enumerate(batch.action):
            action = action.item()
            xs[action].append(batch.state[i])
            rewards[action].append(batch.reward[i])
            weights[action].append(batch.weight[i])

        # learn for each action
        for action, x in xs.items():
            self._linear_regressions[action].train(
                x=torch.stack(x),
                y=torch.tensor(rewards[action]),
                weight=torch.tensor(weights[action]),
            )
        return {}

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        raise NotImplementedError("Integrate with UCB exploration module")
