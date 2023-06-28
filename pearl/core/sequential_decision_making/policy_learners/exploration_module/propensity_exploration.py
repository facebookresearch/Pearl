from typing import Any

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.core.common.policy_learners.exploration_module.value_exploration_base import (
    ValueExplorationBase,
)
from pearl.utils.action_spaces import DiscreteActionSpace


class PropensityExploration(ValueExplorationBase):
    """
    Propensity exploration module.
    """

    def __init__(self) -> None:
        super(PropensityExploration, self).__init__()

    # TODO: We should make discrete action space itself iterable
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        values: torch.Tensor,
        representation: Any = None,
        exploit_action: Action = None,
    ) -> Action:
        return torch.distributions.Categorical(values).sample().item()
