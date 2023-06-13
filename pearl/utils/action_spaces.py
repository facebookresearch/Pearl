import random
from typing import List

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace


class DiscreteActionSpace(ActionSpace):
    def __init__(self, actions: List[Action]):
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
        return len(self.actions[0])

    def to_tensor(self):
        return torch.Tensor(self.actions)
