from abc import ABC, abstractmethod
from enum import Enum

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
        """Resets the internal state of the exploration module. Default implementation does nothing."""
        pass

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action = None,
        # pyre-fixme[9]: values has type `Tensor`; used as `None`.
        values: torch.Tensor = None,
        # pyre-fixme[9]: representation has type `Tensor`; used as `None`.
        representation: torch.Tensor = None,
    ) -> Action:
        pass

    def learn(self, replay_buffer: ReplayBuffer) -> None:  # noqa: B027
        """Learns from the replay buffer. Default implementation does nothing."""
        pass
