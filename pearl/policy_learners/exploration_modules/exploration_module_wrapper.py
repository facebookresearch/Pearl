# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Optional

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules import ExplorationModule
from pearl.replay_buffers.replay_buffer import ReplayBuffer


class ExplorationModuleWrapper(ExplorationModule):
    """
    This class is the base class for all exploration module wrappers.
    """

    def __init__(self, exploration_module: ExplorationModule) -> None:
        self.exploration_module: ExplorationModule = exploration_module

    def reset(self) -> None:  # noqa: B027
        self.exploration_module.reset()

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: Optional[torch.Tensor] = None,
        exploit_action: Optional[Action] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        return self.exploration_module.act(
            subjective_state,
            action_space,
            values,
            exploit_action,
            action_availability_mask,
            representation,
        )

    def learn(self, replay_buffer: ReplayBuffer) -> None:  # noqa: B027
        """Learns from the replay buffer. Default implementation does nothing."""
        self.exploration_module.learn(replay_buffer)
