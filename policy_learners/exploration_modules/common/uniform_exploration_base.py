from abc import abstractmethod

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
    ExplorationType,
)


class UniformExplorationBase(ExplorationModule):
    """
    Uniform exploration module.
    """

    def __init__(self) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self.exploration_type = ExplorationType.UNIFORM

    @abstractmethod
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action,
        # pyre-fixme[9]: values has type `Tensor`; used as `None`.
        values: torch.Tensor = None,
        # pyre-fixme[9]: representation has type `Tensor`; used as `None`.
        representation: torch.Tensor = None,
    ) -> Action:
        pass
