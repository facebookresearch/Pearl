from abc import ABC, abstractmethod

from pearl.api.action import Action


class ActionSpace(ABC):
    """
    An abstract class for action spaces.
    TODO: leverage Gymnasium ones.
    """

    @abstractmethod
    def sample(self) -> Action:
        pass
