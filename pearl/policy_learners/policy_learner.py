from abc import ABC
from typing import Optional

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_module.no_exploration import NoExploration
from pearl.replay_buffer.replay_buffer import ReplayBuffer
from pearl.replay_buffer.transition import TransitionBatch


class PolicyLearner(ABC):
    """
    An abstract interface for policy learners.
    """

    def __init__(
        self, training_rounds: int = 100, batch_size: int = 128, **options
    ) -> None:
        self._exploration_module = (
            options["exploration_module"]
            if "exploration_module" in options
            else NoExploration()
        )
        self._training_rounds = training_rounds
        self._batch_size = batch_size
        self._training_steps = 0

    def reset(self, action_space: ActionSpace) -> None:
        pass

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        exploit_action = self.exploit(subjective_state, available_action_space)

        if exploit:
            return exploit_action
        else:
            return self._exploration_module.act(
                subjective_state, available_action_space, exploit_action
            )

    def exploit(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
    ) -> Action:
        pass

    def learn(
        self, replay_buffer: ReplayBuffer, batch_size: Optional[int] = None
    ) -> None:
        """
        If batch_size is None, use definition from class
        Otherwise, use customized input here
        """
        batch_size = self._batch_size if batch_size is None else batch_size
        if len(replay_buffer) < batch_size:
            return

        for _ in range(self._training_rounds):
            self._training_steps += 1
            batch = replay_buffer.sample(batch_size)
            self.learn_batch(batch)

    def learn_batch(self, batch: TransitionBatch) -> None:
        raise NotImplementedError("learn_batch is not implemented")
