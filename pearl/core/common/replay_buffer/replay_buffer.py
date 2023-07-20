from abc import ABC
from typing import Any, Iterable


class ReplayBuffer(ABC):
    def __init__(self, **options) -> None:
        pass

    def push(self, *args) -> None:
        """Saves a transition."""
        pass

    # TODO: we need to redefine transitions as on-policy algorithms cannot use
    # transitions to learn
    def sample(self, batch_size) -> Iterable[Any]:
        pass

    def empty(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __str__(self):
        return self.__class__.__name__
