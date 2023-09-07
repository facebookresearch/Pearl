from abc import abstractmethod

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
    ExplorationType,
)


class UniformExplorationBase(ExplorationModule):
    """
    Uniform exploration module.
    """

    def __init__(self) -> None:
        self.exploration_type = ExplorationType.UNIFORM

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action,
        values: torch.Tensor = None,
        representation: torch.Tensor = None,
    ) -> Action:
        pass
