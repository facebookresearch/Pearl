from typing import Any, Iterable, Optional

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState

from pearl.replay_buffers.replay_buffer import ReplayBuffer


class SingleTransitionReplayBuffer(ReplayBuffer):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, **options) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self._transition = None

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

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def sample(self, batch_size: int) -> Iterable[Any]:
        assert batch_size == 1, "Only batch size 1 is supported"
        return [self._transition]

    def clear(self) -> None:
        raise Exception("Cannot clear SingleTransitionReplayBuffer")

    def __len__(self) -> int:
        return 1
