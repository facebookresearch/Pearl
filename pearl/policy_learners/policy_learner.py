# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
    SubjectiveState,
)
from pearl.history_summarization_modules.identity_history_summarization_module import (
    IdentityHistorySummarizationModule,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.device import is_distribution_enabled
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class PolicyLearner(torch.nn.Module, ABC):
    """
    An abstract interface for policy learners.

    Important requirements for policy learners using tensors:
        1. Attribute `requires_tensors` must be `True` (this is the default).
        2. If a policy learner is to operate on a given torch device,
           the policy learner must be moved to that device using method `to(device)`.
        3. All inputs to policy leaners must be moved to the proper device,
           including `TransitionBatch`es (which also have a `to(device)` method).
    """

    # See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use  # noqa E501
    # of `T` to annotate `self`. At least one method of `PolicyLearner`
    # returns `self` and we want those return values to be
    # the type of the subclass, not the looser type of `PolicyLearner`.
    T = TypeVar("T", bound="PolicyLearner")

    def __init__(
        self,
        on_policy: bool,
        is_action_continuous: bool,
        action_space: Optional[ActionSpace] = None,
        training_rounds: int = 100,
        batch_size: int = 1,
        requires_tensors: bool = True,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        **options: Any,
    ) -> None:
        super(PolicyLearner, self).__init__()

        self._exploration_module: ExplorationModule = (
            options.get("exploration_module", None) or NoExploration()
        )

        # User needs to either provide the action space or an action representation module at
        # policy learner's initialization for sequential decision making.
        if action_representation_module is None:
            if action_space is not None:
                # If a policy learner is initialized with an action space, then we assume that
                # the agent does not need dynamic action space support.
                self._action_representation_module: ActionRepresentationModule = (
                    IdentityActionRepresentationModule(
                        max_number_actions=action_space.n
                        if isinstance(action_space, DiscreteActionSpace)
                        else -1,
                        representation_dim=action_space.action_dim,
                    )
                )
            else:
                # This is only used in the case of bandit learning applications.
                # TODO: add action representation module for bandit learning applications.
                self._action_representation_module = (
                    IdentityActionRepresentationModule()
                )
        else:
            # User needs to at least specify action dimensions if no action_space is provided.
            assert action_representation_module.representation_dim != -1
            self._action_representation_module = action_representation_module

        self._history_summarization_module: HistorySummarizationModule = (
            IdentityHistorySummarizationModule()
        )

        self._training_rounds = training_rounds
        self._batch_size = batch_size
        self._training_steps = 0
        self.on_policy = on_policy
        self.is_action_continuous = is_action_continuous
        self.distribution_enabled: bool = is_distribution_enabled()
        self.requires_tensors = requires_tensors

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def exploration_module(self) -> ExplorationModule:
        return self._exploration_module

    @property
    def action_representation_module(self) -> ActionRepresentationModule:
        return self._action_representation_module

    @exploration_module.setter
    def exploration_module(self, new_exploration_module: ExplorationModule) -> None:
        self._exploration_module = new_exploration_module

    def get_action_representation_module(self) -> ActionRepresentationModule:
        return self._action_representation_module

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        self._history_summarization_module = value

    def reset(self, action_space: ActionSpace) -> None:
        """Resets policy maker for a new episode. Default implementation does nothing."""
        pass

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        pass

    def learn(
        self,
        replay_buffer: ReplayBuffer,
    ) -> Dict[str, Any]:
        """
        Args:
            replay_buffer: buffer instance which learn is reading from

        Returns:
            A dictionary which includes useful metrics
        """
        batch_size = self._batch_size if not self.on_policy else len(replay_buffer)

        if len(replay_buffer) < batch_size or len(replay_buffer) == 0:
            return {}

        report = {}
        for _ in range(self._training_rounds):
            self._training_steps += 1
            batch = replay_buffer.sample(batch_size)
            single_report = {}
            if isinstance(batch, TransitionBatch):
                batch = self.preprocess_batch(batch)
                single_report = self.learn_batch(batch)

            for k, v in single_report.items():
                if k in report:
                    report[k].append(v)
                else:
                    report[k] = [v]
        return report

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """
        Processes a batch of transitions before passing it to learn_batch().
        This function can be used to implement preprocessing steps such as
        transform the actions.
        """
        batch.state = self._history_summarization_module(batch.state)
        with torch.no_grad():
            batch.next_state = self._history_summarization_module(batch.next_state)

        batch.action = self._action_representation_module(batch.action)
        if batch.next_action is not None:
            batch.next_action = self._action_representation_module(batch.next_action)
        if batch.curr_available_actions is not None:
            batch.curr_available_actions = self._action_representation_module(
                batch.curr_available_actions
            )
        if batch.next_available_actions is not None:
            batch.next_available_actions = self._action_representation_module(
                batch.next_available_actions
            )
        return batch

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Args:
            batch: batch of data that agent is learning from

        Returns:
            A dictionary which includes useful metrics
        """
        raise NotImplementedError("learn_batch is not implemented")

    def __str__(self) -> str:
        return self.__class__.__name__


class DistributionalPolicyLearner(PolicyLearner):
    """
    An abstract interface for distributional policy learners.
    Enforces the property of a risk sensitive safety module.
    """

    def __init__(
        self,
        on_policy: bool,
        is_action_continuous: bool,
        training_rounds: int = 100,
        batch_size: int = 1,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        **options: Any,
    ) -> None:

        super(DistributionalPolicyLearner, self).__init__(
            on_policy=on_policy,
            is_action_continuous=is_action_continuous,
            training_rounds=training_rounds,
            batch_size=batch_size,
            action_representation_module=action_representation_module,
            **options,
        )
