# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.api.reward import Value
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

DEFAULT_ACTION_SPACE = DiscreteActionSpace([torch.tensor([0])])


class ContextualBanditBase(PolicyLearner):
    """
    A base class for Contextual Bandit policy learner.
    """

    def __init__(
        self,
        feature_dim: int,
        exploration_module: Optional[ExplorationModule] = None,
        training_rounds: int = 100,
        batch_size: int = 128,
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        super(ContextualBanditBase, self).__init__(
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            on_policy=False,
            is_action_continuous=False,  # TODO change in subclasses when we add CB for continuous
            action_representation_module=action_representation_module,
        )
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        pass

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        action_availability_mask: Optional[torch.Tensor] = None,
        exploit: bool = False,
    ) -> Action:
        pass

    @abstractmethod
    def get_scores(
        self,
        subjective_state: SubjectiveState,
    ) -> Value:
        """
        Returns:
            Return scores trained by this contextual bandit algorithm
        """
        pass
