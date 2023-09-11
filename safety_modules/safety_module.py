from abc import ABC, abstractmethod

from pearl.api.action_space import ActionSpace
from pearl.core.common.replay_buffer.replay_buffer import ReplayBuffer
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)


class SafetyModule(ABC):
    """
    An abstract interface for exploration module.
    """

    @abstractmethod
    def filter_action(self, subjective_state: SubjectiveState) -> ActionSpace:
        pass

    @abstractmethod
    def learn(self, replay_buffer: ReplayBuffer) -> None:
        pass

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> None:
        pass
