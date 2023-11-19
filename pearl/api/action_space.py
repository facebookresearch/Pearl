from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pearl.api.action import Action
from torch import Tensor


class ActionSpace(ABC):
    """An abstract base class for action spaces. An `ActionSpace` represents the set of
    all possible actions that can be taken by the agent. An `ActionSpace` should be
    specified such that each action is in its "environment representation," which means
    that an `action` from the space can be directly passed to `env.step(action)`.
    """

    @abstractmethod
    def sample(self, mask: Optional[Tensor] = None) -> Action:
        """Sample an action from this space."""
        pass

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Checks whether this is a continuous action space."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Returns the dimensionality of an `Action` element from this space."""
        pass

    @property
    def n(self) -> int:
        """Returns the number of actions in this action space. This will typically
        be implemented for discrete action spaces only."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `n` property."
        )

    @property
    def low(self) -> Tensor:
        """Returns the lower bound of the action space. This will typically be
        implemented for continuous action spaces only."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `low` property."
        )

    @property
    def high(self) -> Tensor:
        """Returns the upper bound of the action space. This will typically be
        implemented for continuous action spaces only."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the `high` property."
        )
