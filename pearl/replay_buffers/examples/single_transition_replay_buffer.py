from typing import List, Optional, Tuple

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState

from pearl.replay_buffers.replay_buffer import ReplayBuffer


# Preferred to define inside class but that is not working
# See https://fb.workplace.com/groups/pyreqa/permalink/7039029492853489/
SingleTransition = Tuple[
    SubjectiveState,
    Action,
    Reward,
    SubjectiveState,
    ActionSpace,
    ActionSpace,
    ActionSpace,
    bool,
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
        action_space: ActionSpace,
        done: bool,
        cost: Optional[float] = None,
    ) -> None:
        self._transition = (
            state,
            action,
            reward,
            next_state,
            curr_available_actions,
            next_available_actions,
            action_space,
            done,
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
