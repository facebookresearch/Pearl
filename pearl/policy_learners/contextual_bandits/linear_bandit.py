#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Any, Dict, Optional

import torch
from pearl.api.action import Action
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.contextual_bandits.contextual_bandit_base import (
    ContextualBanditBase,
    DEFAULT_ACTION_SPACE,
)
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.action_utils import (
    concatenate_actions_to_state,
)
from pearl.utils.functional_utils.learning.linear_regression import LinearRegression
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class LinearBandit(ContextualBanditBase):
    """
    Policy Learner for Contextual Bandit with Linear Policy
    """

    def __init__(
        self,
        feature_dim: int,
        exploration_module: Optional[ExplorationModule] = None,
        l2_reg_lambda: float = 1.0,
        training_rounds: int = 100,
        batch_size: int = 128,
    ) -> None:
        super(LinearBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            # pyre-fixme[6]: For 4th argument expected `ExplorationModule` but got
            #  `Optional[ExplorationModule]`.
            exploration_module=exploration_module,
        )
        self.model = LinearRegression(
            feature_dim=feature_dim, l2_reg_lambda=l2_reg_lambda
        )

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        A <- A + x*x.t
        b <- b + r*x
        """
        x = torch.cat([batch.state, batch.action], dim=1)
        self.model.learn_batch(
            x=x,
            y=batch.reward,
            # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            weight=batch.weight,
        )
        current_values = self.model(x)
        return {"current_values": current_values.mean().item()}

    # pyre-fixme[14]: `act` overrides method defined in `ContextualBanditBase`
    #  inconsistently.
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        action_availability_mask: Optional[torch.Tensor] = None,
        exploit: bool = False,
    ) -> Action:
        """
        Args:
            subjective_state: state will be applied to different action vectors in action_space
            available_action_space: contains a list of action vectors, currenly only supports static space
        Return:
            action index chosen given state and action vectors
        """
        # It doesnt make sense to call act if we are not working with action vector
        assert (
            self._exploration_module is not None
        ), "exploration module must be set to call act()"
        action_count = available_action_space.n
        new_feature = concatenate_actions_to_state(
            subjective_state=subjective_state, action_space=available_action_space
        )
        values = self.model(new_feature)  # (batch_size, action_count)
        assert values.shape == (new_feature.shape[0], action_count)
        return self._exploration_module.act(
            # TODO we might want to name this new_feature
            # so exploration module doesnt need to worry about broadcast state to different action vector
            subjective_state=new_feature,
            action_space=available_action_space,
            values=values,
            action_availability_mask=action_availability_mask,
            representation=self.model,
        )

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace = DEFAULT_ACTION_SPACE,
    ) -> torch.Tensor:
        """
        Returns:
            UCB scores when exploration module is UCB
            Shape is (batch)
        """
        assert isinstance(self._exploration_module, ScoreExplorationBase)
        feature = concatenate_actions_to_state(
            subjective_state=subjective_state, action_space=action_space
        )
        return self._exploration_module.get_scores(
            subjective_state=feature,
            values=self.model(feature),
            action_space=action_space,
            representation=self.model,
        ).squeeze()
