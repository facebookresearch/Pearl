from abc import ABC

from pearl.api.action import Action


class ActionSpace(ABC):
    """
    An abstract class for action spaces.
    TODO: leverage Gymnasium ones.
    """

    def __init__(self):
        pass

    def sample(self) -> Action:
        pass
