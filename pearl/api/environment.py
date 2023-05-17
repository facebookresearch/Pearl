from abc import ABC, abstractmethod

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation


class Environment(ABC):
    """
    An abstract interface for environments.
    """

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        pass

    def reset(self) -> (Observation, ActionSpace):
        pass

    def step(self, action: Action) -> ActionResult:
        pass

    def render(self):
        pass

    def close(self):
        pass
