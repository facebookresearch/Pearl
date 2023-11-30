from abc import ABC, abstractmethod

from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.policy_learner import PolicyLearner

from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch


class SafetyModule(ABC):
    """
    An abstract interface for exploration module.
    """

    @abstractmethod
    def filter_action(self, subjective_state: SubjectiveState) -> ActionSpace:
        pass

    @abstractmethod
    def reset(self, action_space: ActionSpace) -> None:
        pass

    @abstractmethod
    def learn(self, replay_buffer: ReplayBuffer, policy_learner: PolicyLearner) -> None:
        pass

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> None:
        pass
