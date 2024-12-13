# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import random
from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
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
