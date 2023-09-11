from abc import ABC, abstractmethod

from pearl.api.observation import Observation

from pearl.api.state import SubjectiveState
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch


class HistorySummarizationModule(ABC):
    """
    An abstract interface for exploration module.
    """

    @abstractmethod
    def summarize_history(
        self, subjective_state: SubjectiveState, observation: Observation
    ) -> SubjectiveState:
        pass

    @abstractmethod
    def learn(self, replay_buffer: ReplayBuffer) -> None:
        pass

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> None:
        raise NotImplementedError("learn_batch is not implemented")
