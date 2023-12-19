# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod

from Pearl.pearl.api.action_space import ActionSpace
from Pearl.pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from Pearl.pearl.policy_learners.policy_learner import PolicyLearner

from Pearl.pearl.replay_buffers.replay_buffer import ReplayBuffer
from Pearl.pearl.replay_buffers.transition import TransitionBatch


class SafetyModule(ABC):
    """
    An abstract interface for exploration module.
    """

    @abstractmethod
    def filter_action(
        self, subjective_state: SubjectiveState, action_space: ActionSpace
    ) -> ActionSpace:
        pass

    @abstractmethod
    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        pass

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> None:
        pass
