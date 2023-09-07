import random

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.core.common.policy_learners.exploration_module.uniform_exploration_base import (
    UniformExplorationBase,
)


class EGreedyExploration(UniformExplorationBase):
    """
    epsilon Greedy exploration module.
    """

    def __init__(self, epsilon: float) -> None:
        super(EGreedyExploration, self).__init__()
        self.epsilon = epsilon

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action,
        values: torch.Tensor = None,
        representation: torch.Tensor = None,
    ) -> Action:
        if random.random() < self.epsilon:
            return action_space.sample()

        return exploit_action
