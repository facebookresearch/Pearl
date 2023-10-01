from typing import Any

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.common.value_exploration_base import (
    ValueExplorationBase,
)
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class PropensityExploration(ValueExplorationBase):
    """
    Propensity exploration module.
    """

    def __init__(self) -> None:
        super(PropensityExploration, self).__init__()

    # TODO: We should make discrete action space itself iterable
    # pyre-fixme[14]: `act` overrides method defined in `ValueExplorationBase`
    #  inconsistently.
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        values: torch.Tensor,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        representation: Any = None,
        exploit_action: Action = None,
    ) -> Action:
        return torch.distributions.Categorical(values).sample().item()
