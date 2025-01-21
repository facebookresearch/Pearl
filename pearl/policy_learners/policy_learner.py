# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, List, TypeVar

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
        action_space: ActionSpace | None = None,
        training_rounds: int = 100,
        batch_size: int = 1,
        requires_tensors: bool = True,
        action_representation_module: ActionRepresentationModule | None = None,
        **options: Any,
    ) -> None:
        super().__init__()

        exploration_module = options.get("exploration_module", None) or NoExploration()
        self.exploration_module = exploration_module

        # User needs to either provide the action space or an action representation module at
        # policy learner's initialization for sequential decision making.
        if action_representation_module is None:
            if action_space is not None:
                # If a policy learner is initialized with an action space, then we assume that
                # the agent does not need dynamic action space support.
                action_representation_module = IdentityActionRepresentationModule(
                    max_number_actions=(
                        action_space.n
                        if isinstance(action_space, DiscreteActionSpace)
                        else None
                    ),
                    representation_dim=action_space.action_dim,
                )
            else:
                # This is only used in the case of bandit learning applications.
                # TODO: add action representation module for bandit learning applications.
                action_representation_module = IdentityActionRepresentationModule()
        else:
            # User needs to at least specify action dimensions if no action_space is provided.
            assert action_representation_module.representation_dim is not None
            action_representation_module = action_representation_module

        self.action_representation_module = action_representation_module

        self._history_summarization_module: HistorySummarizationModule = (
            IdentityHistorySummarizationModule()
        )

        self._training_rounds = training_rounds
        self._batch_size = batch_size
        self._training_steps = 0
        self.on_policy = on_policy
        self._is_action_continuous = is_action_continuous
        self.distribution_enabled: bool = is_distribution_enabled()
        self.requires_tensors = requires_tensors

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def exploration_module(self) -> ExplorationModule:
        exploration_module = self._modules["exploration_module"]
        assert exploration_module is not None
        assert isinstance(exploration_module, ExplorationModule)
        return exploration_module

    # Setter is never executed because nn.Module overrides __setattr__ of modules.
    # This is defined to satisfy the type checker.
    @exploration_module.setter
    def exploration_module(self, value: ExplorationModule) -> None:
        self._modules["exploration_module"] = value

    @property
    def action_representation_module(self) -> ActionRepresentationModule:
        action_representation_module = self._modules["action_representation_module"]
        assert action_representation_module is not None
        assert isinstance(action_representation_module, ActionRepresentationModule)
        return action_representation_module

    # Setter is never executed because nn.Module overrides __setattr__ of modules.
    # This is defined to satisfy the type checker.
    @action_representation_module.setter
    def action_representation_module(self, value: ActionRepresentationModule) -> None:
        self._modules["action_representation_module"] = value

    def get_action_representation_module(self) -> ActionRepresentationModule:
        return self.action_representation_module

    @abstractmethod
    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        pass

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
    ) -> dict[str, Any]:
        """
        Args:
            replay_buffer: buffer instance which learn is reading from

        Returns:
            A dictionary which includes useful metrics
        """
        if len(replay_buffer) == 0:
            return {}

        if self._batch_size == -1 or len(replay_buffer) < self._batch_size:
            batch_size = len(replay_buffer)
        else:
            batch_size = self._batch_size

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

        batch.action = self.action_representation_module(batch.action)
        if batch.next_action is not None:
            batch.next_action = self.action_representation_module(batch.next_action)
        if batch.curr_available_actions is not None:
            batch.curr_available_actions = self.action_representation_module(
                batch.curr_available_actions
            )
        if batch.next_available_actions is not None:
            batch.next_available_actions = self.action_representation_module(
                batch.next_available_actions
            )
        return batch

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        """
        Args:
            batch: batch of data that agent is learning from

        Returns:
            A dictionary which includes useful metrics
        """
        raise NotImplementedError("learn_batch is not implemented")

    def __str__(self) -> str:
        return self.__class__.__name__

    def compare(self, other: "PolicyLearner") -> str:
        """
        Compares two PolicyLearner instances for equality,
        checking attributes and modules.
        Note: subcomponents which are PyTorch modules are
        compared by state dict only.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        if not isinstance(other, PolicyLearner):
            differences.append("other is not an instance of PolicyLearner")
        else:
            # Compare attributes
            if self._training_rounds != other._training_rounds:
                differences.append(
                    f"_training_rounds is different: {self._training_rounds} vs "
                    + f"{other._training_rounds}"
                )
            if self._batch_size != other._batch_size:
                differences.append(
                    f"_batch_size is different: {self._batch_size} vs {other._batch_size}"
                )
            if self.on_policy != other.on_policy:
                differences.append(
                    f"on_policy is different: {self.on_policy} vs {other.on_policy}"
                )
            if self._is_action_continuous != other._is_action_continuous:
                differences.append(
                    f"_is_action_continuous is different: {self._is_action_continuous} vs "
                    + f"{other._is_action_continuous}"
                )

            # Compare exploration modules
            if self.exploration_module is None:
                if other.exploration_module is not None:
                    differences.append(
                        "exploration_module is different: None vs not None"
                    )
            elif (
                reason := self.exploration_module.compare(other.exploration_module)
            ) != "":
                differences.append(f"exploration_module is different: {reason}")

            # Compare action representation modules
            if (
                reason := self.action_representation_module.compare(
                    other.action_representation_module
                )
            ) != "":
                differences.append(
                    f"action_representation_module is different: {reason}"
                )

            # Compare history summarization modules
            if (
                reason := self._history_summarization_module.compare(
                    other._history_summarization_module
                )
            ) != "":
                differences.append(
                    f"history summarization module is different: {reason}"
                )

        return "\n".join(differences)


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
        action_representation_module: ActionRepresentationModule | None = None,
        **options: Any,
    ) -> None:
        super().__init__(
            on_policy=on_policy,
            is_action_continuous=is_action_continuous,
            training_rounds=training_rounds,
            batch_size=batch_size,
            action_representation_module=action_representation_module,
            **options,
        )
