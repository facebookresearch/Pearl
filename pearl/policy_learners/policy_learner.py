from abc import ABC
from typing import Any, Dict, Optional

import torch

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
        pass

    def learn(
        self, replay_buffer: ReplayBuffer, batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Args:
            replay_buffer: buffer instance which learn is reading from
            batch_size: size of data that we would like one round of train to work with
                If batch_size is None, use definition from class
                Otherwise, use customized input here

        Returns:
            A dictionary which includes useful metric to return to upperlevel for different purpose eg debugging
        """
        batch_size = self._batch_size if batch_size is None else batch_size
        if len(replay_buffer) < batch_size:
            return {}

        report = {}
        for _ in range(self._training_rounds):
            self._training_steps += 1
            batch = replay_buffer.sample(batch_size)
            single_report = self.learn_batch(batch)
            for k, v in single_report.items():
                if k in report:
                    report[k].append(v)
                else:
                    report[k] = [v]
        return report

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Args:
            batch: batch of data that agent is learning from

        Returns:
            A dictionary which includes useful metric to return to upperlevel for different purpose eg debugging
        """
        raise NotImplementedError("learn_batch is not implemented")
