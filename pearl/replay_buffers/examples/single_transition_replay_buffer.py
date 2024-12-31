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
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState

from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.utils.device import get_default_device


# Preferred to define inside class but that is not working. Pending discussion.
SingleTransition = tuple[
    SubjectiveState,
    Action,
    Reward,
    SubjectiveState,
    ActionSpace,
    ActionSpace,
    bool,
    bool,
    Optional[int],
    Optional[float],
]


def to_default_device_if_tensor(obj: object) -> object:
    if isinstance(obj, torch.Tensor):
        return obj.to(get_default_device())
    else:
        return obj


class SingleTransitionReplayBuffer(ReplayBuffer):
    def __init__(self) -> None:
        self._transition: SingleTransition | None = None

    @property
    def device_for_batches(self) -> torch.device:
        raise ValueError("SingleTransitionReplayBuffer does not have a device.")

    @device_for_batches.setter
    def device_for_batches(self, new_device_for_batches: torch.device) -> None:
        pass

    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        truncated: bool,
        curr_available_actions: ActionSpace | None = None,
        next_state: SubjectiveState | None = None,
        next_available_actions: ActionSpace | None = None,
        max_number_actions: int | None = None,
        cost: float | None = None,
    ) -> None:
        # TODO: we use pyre-ignore here because tabular Q learning does not use tensors
        # like other policy learners. It should be converted to do so.
        self._transition = (  # pyre-ignore
            to_default_device_if_tensor(state),
            to_default_device_if_tensor(action),
            to_default_device_if_tensor(reward),
            to_default_device_if_tensor(next_state),
            curr_available_actions,
            next_available_actions,
            to_default_device_if_tensor(terminated),
            to_default_device_if_tensor(truncated),
            max_number_actions,
            to_default_device_if_tensor(cost),
        )

    def sample(self, batch_size: int) -> list[SingleTransition]:
        assert batch_size == 1, "Only batch size 1 is supported"
        assert (
            self._transition is not None
        ), "No transition in SingleTransitionReplayBuffer"
        return [self._transition]

    def clear(self) -> None:
        raise Exception("Cannot clear SingleTransitionReplayBuffer")

    def __len__(self) -> int:
        return 1
