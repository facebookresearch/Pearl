# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import random
from typing import Any, List, Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules import ExplorationModule
from pearl.policy_learners.exploration_modules.common.uniform_exploration_base import (
    UniformExplorationBase,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class EGreedyExploration(UniformExplorationBase):
    """
    epsilon Greedy exploration module.
    """

    def __init__(
        self,
        epsilon: float,
        start_epsilon: Optional[float] = None,
        end_epsilon: Optional[float] = None,
        warmup_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.warmup_steps = warmup_steps
        self.time_step = 0
        self._epsilon_scheduling: bool = (
            self.start_epsilon is not None
            and self.end_epsilon is not None
            and self.warmup_steps is not None
        )
        if self._epsilon_scheduling:
            assert self.start_epsilon is not None
            self.curr_epsilon: float = self.start_epsilon
        else:
            self.curr_epsilon = epsilon

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action | None,
        values: torch.Tensor | None = None,
        action_availability_mask: torch.Tensor | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        if self._epsilon_scheduling:
            assert self.warmup_steps is not None
            if self.time_step < self.warmup_steps:
                assert self.warmup_steps is not None
                frac = self.time_step / self.warmup_steps
                assert self.start_epsilon is not None
                assert self.end_epsilon is not None
                self.curr_epsilon = (
                    self.start_epsilon + (self.end_epsilon - self.start_epsilon) * frac
                )
        self.time_step += 1
        if exploit_action is None:
            raise ValueError(
                "exploit_action cannot be None for epsilon-greedy exploration"
            )
        if not isinstance(action_space, DiscreteActionSpace):
            raise TypeError("action space must be discrete")
        if random.random() < self.curr_epsilon:
            return action_space.sample(action_availability_mask).to(
                exploit_action.device
            )
        else:
            return exploit_action

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "start_epsilon": self.start_epsilon,
            "curr_epsilon": self.curr_epsilon,
            "end_epsilon": self.end_epsilon,
            "time_step": self.time_step,
            "warmup_steps": self.warmup_steps,
            "epsilon_scheduling": self._epsilon_scheduling,
        }

    def set_extra_state(self, state: Any) -> None:  # pyre-ignore
        assert isinstance(state, dict)
        self.start_epsilon = state["start_epsilon"]
        self.curr_epsilon = state["curr_epsilon"]
        self.end_epsilon = state["end_epsilon"]
        self.time_step = state["time_step"]
        self.warmup_steps = state["warmup_steps"]
        self._epsilon_scheduling = state["epsilon_scheduling"]

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two EGreedyExploration instances for equality,
        checking attributes.

        Args:
        other: The other ExplorationModule to compare with.

        Returns:
        str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        # Call the compare method of the parent class (UniformExplorationBase)
        differences.append(super().compare(other))

        if not isinstance(other, EGreedyExploration):
            differences.append("other is not an instance of EGreedyExploration")
        else:
            if self.start_epsilon != other.start_epsilon:
                differences.append(
                    f"start_epsilon is different: {self.start_epsilon} vs {other.start_epsilon}"
                )
            if self.end_epsilon != other.end_epsilon:
                differences.append(
                    f"end_epsilon is different: {self.end_epsilon} vs {other.end_epsilon}"
                )
            if self.time_step != other.time_step:
                differences.append(
                    f"time_step is different: {self.time_step} vs {other.time_step}"
                )
            if self._epsilon_scheduling != other._epsilon_scheduling:
                differences.append(
                    f"_epsilon_scheduling is different: {self._epsilon_scheduling} "
                    + f"vs {other._epsilon_scheduling}"
                )
            if self.warmup_steps != other.warmup_steps:
                differences.append(
                    f"warmup_steps is different: {self.warmup_steps} vs {other.warmup_steps}"
                )
            if self.curr_epsilon != other.curr_epsilon:
                differences.append(
                    f"curr_epsilon is different: {self.curr_epsilon} vs {other.curr_epsilon}"
                )

        return "\n".join(differences)
