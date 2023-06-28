from abc import ABC

from pearl.api.action_space import ActionSpace
from pearl.core.common.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.core.common.replay_buffer.replay_buffer import ReplayBuffer
from pearl.core.common.replay_buffer.transition import TransitionBatch


class SafetyModule(ABC):
    """
    An abstract interface for exploration module.
    """

    def __init__(self, **options) -> None:
        pass

    def filter_action(self, subjective_state: SubjectiveState) -> ActionSpace:
        pass

    def learn(self, replay_buffer: ReplayBuffer) -> None:
        pass

    def learn_batch(self, batch: TransitionBatch) -> None:
        raise NotImplementedError("learn_batch is not implemented")
