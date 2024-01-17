# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action import Action
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.contextual_bandit.linear_regression import LinearRegression
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
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        super(LinearBandit, self).__init__(
            feature_dim=feature_dim,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            action_representation_module=action_representation_module,
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
        assert batch.weight is not None
        self.model.learn_batch(
            x=x,
            y=batch.reward,
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
            available_action_space: contains a list of action vectors.
                                    Currently, only static spaces are supported.
        Return:
            action index chosen given state and action vectors
        """
        # It doesnt make sense to call act if we are not working with action vector
        assert (
            self._exploration_module is not None
        ), "exploration module must be set to call act()"
        action_count = available_action_space.n
        new_feature = concatenate_actions_to_state(
            subjective_state=subjective_state,
            action_space=available_action_space,
            action_representation_module=self._action_representation_module,
        )
        values = self.model(new_feature)  # (batch_size, action_count)
        assert values.shape == (new_feature.shape[0], action_count)
        return self._exploration_module.act(
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
            subjective_state=subjective_state,
            action_space=action_space,
            action_representation_module=self._action_representation_module,
        )
        assert isinstance(self._exploration_module, ScoreExplorationBase)
        return self._exploration_module.get_scores(
            subjective_state=feature,
            values=self.model(feature),
            action_space=action_space,
            representation=self.model,
        ).squeeze()
