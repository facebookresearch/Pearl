import random
from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.common.uniform_exploration_base import (
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
        values: Optional[torch.Tensor] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        if action_availability_mask is not None:
            raise NotImplementedError("Action availability mask is not supported.")
        if random.random() < self.epsilon:
            return action_space.sample()

        return exploit_action
