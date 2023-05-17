from abc import ABC

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.replay_buffer.replay_buffer import ReplayBuffer


class ExplorationModule(ABC):
    """
    An abstract interface for exploration module.
    """

    def __init__(self, **options) -> None:
        pass

    def reset(self) -> None:
        pass

    def act(
        self, subjective_state: SubjectiveState, action_space: ActionSpace
    ) -> Action:
        pass

    def learn(self, replay_buffer: ReplayBuffer) -> None:
        pass
