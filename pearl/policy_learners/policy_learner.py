from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.neural_networks.optimizers.keyed_optimizer_wrapper import NoOpOptimizer
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.device import is_distribution_enabled


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
        training_rounds: int = 100,
        batch_size: int = 1,
        requires_tensors: bool = True,
        # temporary solution before abstract interfaces are implemented
        # pyre-fixme[2]: Parameter must be annotated.
        **options,
    ) -> None:
        super(PolicyLearner, self).__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self._exploration_module = (
            options["exploration_module"]
            if "exploration_module" in options
            else NoExploration()
        )
        self._training_rounds = training_rounds
        self._batch_size = batch_size
        self._training_steps = 0
        self.on_policy = on_policy
        self.is_action_continuous = is_action_continuous
        self.distribution_enabled: bool = is_distribution_enabled()
        self.requires_tensors = requires_tensors

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return NoOpOptimizer()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def exploration_module(self) -> ExplorationModule:
        return self._exploration_module

    @exploration_module.setter
    # pyre-fixme[2]: Parameter must be annotated.
    def exploration_module(self, new_exploration_module) -> None:
        self._exploration_module = new_exploration_module

    def reset(self, action_space: ActionSpace) -> None:  # noqa: B027
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
            A dictionary which includes useful metric to return to upperlevel for different purpose eg debugging
        """
        batch_size = self._batch_size if not self.on_policy else len(replay_buffer)

        if len(replay_buffer) < batch_size or len(replay_buffer) == 0:
            # logging.warning(
            #     f"Batch size is {batch_size} and replay buffer size is {len(replay_buffer)}; we don't have enough data to learn."
            # )
            return {}

        report = {}
        for _ in range(self._training_rounds):
            self._training_steps += 1
            batch = replay_buffer.sample(batch_size)
            # pyre-fixme[6]: For 1st argument expected `TransitionBatch` but got
            #  `Iterable[typing.Any]`.
            single_report = self.learn_batch(batch)
            for k, v in single_report.items():
                if k in report:
                    report[k].append(v)
                else:
                    report[k] = [v]
        return report

    @abstractmethod
    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Args:
            batch: batch of data that agent is learning from

        Returns:
            A dictionary which includes useful metric to return to upperlevel for different purpose eg debugging
        """
        raise NotImplementedError("learn_batch is not implemented")

    # pyre-fixme[3]: Return type must be annotated.
    def __str__(self):
        return self.__class__.__name__


class DistributionalPolicyLearner(PolicyLearner):
    """
    An abstract interface for distributional policy learners. Enforces the property of a risk sensitive safety module.
    """

    def __init__(
        self,
        on_policy: bool,
        is_action_continuous: bool,
        training_rounds: int = 100,
        batch_size: int = 1,
        # pyre-fixme[2]: Parameter must be annotated.
        **options,
    ) -> None:

        super(DistributionalPolicyLearner, self).__init__(
            on_policy=on_policy,
            is_action_continuous=is_action_continuous,
            training_rounds=training_rounds,
            batch_size=batch_size,
            **options,
        )
