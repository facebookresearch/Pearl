# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.exploration_modules.exploration_module_wrapper import (
    ExplorationModuleWrapper,
)


class Warmup(ExplorationModuleWrapper):
    """
    Follow the random policy for the first `warmup_steps` steps,
    then switch to the actions from the base exploration module.
    """

    def __init__(
        self, exploration_module: ExplorationModule, warmup_steps: int
    ) -> None:
        self.warmup_steps = warmup_steps
        self.time_step = 0
        super().__init__(exploration_module)

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor | None = None,
        exploit_action: Action | None = None,
        action_availability_mask: torch.Tensor | None = None,
        representation: torch.nn.Module | None = None,
    ) -> Action:
        if self.time_step < self.warmup_steps:
            action = action_space.sample()
        else:
            action = self.exploration_module.act(
                subjective_state=subjective_state,
                action_space=action_space,
                values=values,
                exploit_action=exploit_action,
                action_availability_mask=action_availability_mask,
                representation=representation,
            )
        self.time_step += 1
        return action

    def compare(self, other: ExplorationModule) -> str:
        """
        Compares two Warmup instances for equality,
        checking attributes and the underlying exploration module.

        Args:
          other: The other ExplorationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, Warmup):
            differences.append("other is not an instance of Warmup")
        else:
            if self.warmup_steps != other.warmup_steps:
                differences.append(
                    f"warmup_steps is different: {self.warmup_steps} vs {other.warmup_steps}"
                )
            if self.time_step != other.time_step:
                differences.append(
                    f"time_step is different: {self.time_step} vs {other.time_step}"
                )

            # Compare the underlying exploration modules
            if (
                reason := self.exploration_module.compare(other.exploration_module)
            ) != "":
                differences.append(f"exploration_module is different: {reason}")

        return "\n".join(differences)
