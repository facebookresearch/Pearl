import random
from typing import List

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace


class DiscreteActionSpace(ActionSpace):
    def __init__(self, actions: List[Action]):
        """
        actions: List[Action] could be a list of action vector
        or a range of action index
        TODO better idea to write this cleaner?
        """
        self.actions = actions
        self.n = len(actions)

    def sample(self) -> Action:
        return random.choice(self.actions)

    def __iter__(self) -> Action:
        for action in self.actions:
            yield action

    def __getitem__(self, index):
        return self.actions[index]

    @property
    def action_dim(self):
        try:
            return len(self.actions[0])
        except TypeError:
            return 0  # indicate that init with action index

    def to_tensor(self):
        if self.action_dim == 0:
            return torch.zeros(self.n, 0)
        return torch.Tensor(self.actions)
