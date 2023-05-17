from abc import ABC

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation


class Agent(ABC):
    """
    An abstract interface for agents.
    """

    def act(self) -> Action:
        pass

    def observe(self, action_result: ActionResult) -> None:
        pass

    def learn(self) -> None:
        pass

    def reset(self, observation: Observation, action_space: ActionSpace) -> None:
        pass
