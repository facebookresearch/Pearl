import random
from typing import List

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace


class DiscreteActionSpace(ActionSpace):
    def __init__(self, actions: List[Action]):
        self.actions = actions
        self.n = len(actions)

    def sample(self) -> Action:
        return random.choice(self.actions)
