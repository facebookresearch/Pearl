# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer


class ExplorationType(Enum):
    UNIFORM = 0
    BOLTZMANN = 1
    REPRESENTATION = 2
    EPISTEMICNN = 3
    VALUE = 4


class ExplorationModule(ABC):
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
        values: Optional[torch.Tensor] = None,
        exploit_action: Optional[Action] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        pass

    def learn(self, replay_buffer: ReplayBuffer) -> None:  # noqa: B027
        """Learns from the replay buffer. Default implementation does nothing."""
        pass
