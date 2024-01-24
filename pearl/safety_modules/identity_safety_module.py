# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional

from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.safety_modules.safety_module import SafetyModule


class IdentitySafetyModule(SafetyModule):
    """
    A safety module that does not restrict action spaces.
    """

    def filter_action(
        self, subjective_state: SubjectiveState, action_space: ActionSpace
    ) -> ActionSpace:
        return action_space

    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        pass

    def learn_batch(self, batch: TransitionBatch) -> None:
        pass
