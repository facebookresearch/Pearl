from abc import ABC
from typing import Any

from pearl.api.observation import Observation

from pearl.api.state import SubjectiveState
from pearl.replay_buffer.replay_buffer import ReplayBuffer
from pearl.replay_buffer.transition import TransitionBatch


class HistorySummarizationModule(ABC):
    """
    An abstract interface for exploration module.
    """

    def __init__(self, **options) -> None:
        pass

    def summarize_history(
        self, subjective_state: SubjectiveState, observation: Observation
    ) -> SubjectiveState:
        pass

    def learn(self, replay_buffer: ReplayBuffer) -> None:
        pass

    def learn_batch(self, batch: TransitionBatch) -> None:
        raise NotImplementedError("learn_batch is not implemented")
