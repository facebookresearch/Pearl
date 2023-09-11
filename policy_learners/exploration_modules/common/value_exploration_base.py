from abc import abstractmethod

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
    ExplorationType,
)
from pearl.utils.device import get_pearl_device


class ValueExplorationBase(ExplorationModule):
    """
    Value exploration module.
    """

    def __init__(self) -> None:
        self.exploration_type = ExplorationType.VALUE
        self.device = get_pearl_device()

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        exploit_action: Action = None,
        representation: torch.Tensor = None,
    ) -> Action:
        pass
