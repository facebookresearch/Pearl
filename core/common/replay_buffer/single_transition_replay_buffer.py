from typing import Any, Iterable

from pearl.core.common.replay_buffer.replay_buffer import ReplayBuffer


class SingleTransitionReplayBuffer(ReplayBuffer):
    def __init__(self, **options) -> None:
        self._transition = None

    def push(self, *args) -> None:
        self._transition = args

    def sample(self, batch_size) -> Iterable[Any]:
        assert batch_size == 1, "Only batch size 1 is supported"
        return [self._transition]

    def __len__(self) -> int:
        return 1
