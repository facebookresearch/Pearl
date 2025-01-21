# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import ABC, abstractmethod
from enum import Enum

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from torch import nn


class ExplorationType(Enum):
    UNIFORM = 0
    BOLTZMANN = 1
    REPRESENTATION = 2
    EPISTEMICNN = 3
    VALUE = 4


class ExplorationModule(ABC, nn.Module):
    """
    An abstract interface for exploration module.
    """

    def reset(self) -> None:  # noqa: B027
        """
        Resets the internal state of the exploration module. Default implementation does nothing.
        """
        pass

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor | None = None,
        exploit_action: Action | None = None,
        action_availability_mask: torch.Tensor | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        pass

    def learn(self, replay_buffer: ReplayBuffer) -> None:  # noqa: B027
        """Learns from the replay buffer. Default implementation does nothing."""
        pass

    @abstractmethod
    def compare(self, other: "ExplorationModule") -> str:
        """
        Compares two exploration modules.
        Note: subcomponents which are PyTorch modules are
        compared by state dict only.
        Args:
            other: The other exploration module to compare with.
        Returns:
            A string describing the differences, or an empty string if they are identical.
        """
        ...
