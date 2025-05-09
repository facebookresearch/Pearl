# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import ABC, abstractmethod

import torch

from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.policy_learner import PolicyLearner

from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch


class SafetyModule(torch.nn.Module, ABC):
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

    @abstractmethod
    def compare(self, other: "SafetyModule") -> str:
        """
        Compares two SafetyModule instances for equality.
        Note: subcomponents which are PyTorch modules are
        compared by state dict only.

        Args:
            other: another SafetyModule instance to compare with
        Returns:
            str: a string describing the differences,
            or an empty string if they are identical.
        """
        pass
