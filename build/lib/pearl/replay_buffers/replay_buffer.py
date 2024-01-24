# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Optional

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState


class ReplayBuffer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._is_action_continuous: bool = False
        self._has_cost_available: bool = False

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @device.setter
    @abstractmethod
    def device(self, new_device: torch.device) -> None:
        pass

    @abstractmethod
    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        next_state: SubjectiveState,
        curr_available_actions: ActionSpace,
        next_available_actions: ActionSpace,
        done: bool,
        max_number_actions: Optional[int],
        cost: Optional[float] = None,
    ) -> None:
        """Saves a transition."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> object:
        pass

    @abstractmethod
    def clear(self) -> None:
        """Empties replay buffer"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    @property
    def is_action_continuous(self) -> bool:
        """Whether the action space is continuous or not."""
        return self._is_action_continuous

    @is_action_continuous.setter
    def is_action_continuous(self, value: bool) -> None:
        """Set whether the action space is continuous or not."""
        self._is_action_continuous = value

    @property
    def has_cost_available(self) -> bool:
        return self._has_cost_available

    @has_cost_available.setter
    def has_cost_available(self, value: bool) -> None:
        self._has_cost_available = value
