from abc import ABC, abstractmethod
from typing import Any, Iterable


class ReplayBuffer(ABC):
    @abstractmethod
    def push(self, *args) -> None:
        """Saves a transition."""
        pass

    # TODO: we need to redefine transitions as on-policy algorithms cannot use
    # transitions to learn
    @abstractmethod
    def sample(self, batch_size) -> Iterable[Any]:
        pass

    def empty(self) -> None:
        """Indicates whether replay buffer is empty. Default implementations is len(self) == 0."""
        return len(self) == 0

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __str__(self):
        return self.__class__.__name__
