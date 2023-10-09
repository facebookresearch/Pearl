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
from pearl.utils.device import get_pearl_device


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

    def __init__(self) -> None:
        self.device: torch.device = get_pearl_device()

    def reset(self) -> None:  # noqa: B027
        """Resets the internal state of the exploration module. Default implementation does nothing."""
        pass

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: Optional[torch.Tensor] = None,
        exploit_action: Action = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        pass

    def learn(self, replay_buffer: ReplayBuffer) -> None:  # noqa: B027
        """Learns from the replay buffer. Default implementation does nothing."""
        pass
