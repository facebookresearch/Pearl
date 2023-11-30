from typing import Any, Iterable

import torch

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

    # pyre-fixme[2]: Parameter must be annotated.
    def push(self, *args) -> None:
        self._transition = args

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def sample(self, batch_size: int) -> Iterable[Any]:
        assert batch_size == 1, "Only batch size 1 is supported"
        return [self._transition]

    def clear(self) -> None:
        raise Exception("Cannot clear SingleTransitionReplayBuffer")

    def __len__(self) -> int:
        return 1
