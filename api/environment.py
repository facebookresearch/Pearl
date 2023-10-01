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

    @abstractmethod
    # pyre-fixme[31]: Expression `ActionSpace)` is not a valid type.
    def reset(self) -> (Observation, ActionSpace):
        pass

    @abstractmethod
    def step(self, action: Action) -> ActionResult:
        pass

    # pyre-fixme[3]: Return type must be annotated.
    def render(self):  # noqa: B027
        """Renders environment. Default implementation does nothing."""
        pass

    # pyre-fixme[3]: Return type must be annotated.
    def close(self):  # noqa: B027
        """Closes environment, taking care of any cleanup needed. Default implementation does nothing."""
        pass
