from abc import ABC, abstractmethod
from typing import Any, Iterable


class ReplayBuffer(ABC):
    def __init__(self) -> None:
        self._is_action_continuous: bool = False

    @abstractmethod
    # pyre-fixme[2]: Parameter must be annotated.
    def push(self, *args) -> None:
        """Saves a transition."""
        pass

    # TODO: we need to redefine transitions as on-policy algorithms cannot use
    # transitions to learn
    @abstractmethod
    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    # pyre-fixme[2]: Parameter must be annotated.
    def sample(self, batch_size) -> Iterable[Any]:
        pass

    def empty(self) -> None:
        """Indicates whether replay buffer is empty. Default implementations is len(self) == 0."""
        # pyre-fixme[7]: Expected `None` but got `bool`.
        return len(self) == 0

    @abstractmethod
    def __len__(self) -> int:
        pass

    # pyre-fixme[3]: Return type must be annotated.
    def __str__(self):
        return self.__class__.__name__

    @property
    def is_action_continuous(self) -> bool:
        """Whether the action space is continuous or not."""
        return self._is_action_continuous

    @is_action_continuous.setter
    def is_action_continuous(self, value: bool) -> None:
        """Set whether the action space is continuous or not."""
        self._is_action_continuous = value
