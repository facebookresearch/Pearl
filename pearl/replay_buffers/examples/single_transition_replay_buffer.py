# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Tuple

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState

from pearl.replay_buffers.replay_buffer import ReplayBuffer


# Preferred to define inside class but that is not working. Pending discussion.
SingleTransition = Tuple[
    SubjectiveState,
    Action,
    Reward,
    SubjectiveState,
    ActionSpace,
    ActionSpace,
    bool,
    Optional[int],
    Optional[float],
]


class SingleTransitionReplayBuffer(ReplayBuffer):
    def __init__(self) -> None:
        self._transition: Optional[SingleTransition] = None

    @property
    def device(self) -> torch.device:
        raise ValueError("SingleTransitionReplayBuffer does not have a device.")

    @device.setter
    def device(self, new_device: torch.device) -> None:
        pass

    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        next_state: SubjectiveState,
        curr_available_actions: ActionSpace,
        next_available_actions: ActionSpace,
        done: bool,
        max_number_actions: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> None:
        self._transition = (
            state,
            action,
            reward,
            next_state,
            curr_available_actions,
            next_available_actions,
            done,
            max_number_actions,
            cost,
        )

    def sample(self, batch_size: int) -> List[SingleTransition]:
        assert batch_size == 1, "Only batch size 1 is supported"
        assert (
            self._transition is not None
        ), "No transition in SingleTransitionReplayBuffer"
        return [self._transition]

    def clear(self) -> None:
        raise Exception("Cannot clear SingleTransitionReplayBuffer")

    def __len__(self) -> int:
        return 1
