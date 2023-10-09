from typing import Any, Optional

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


class PropensityExploration(ExplorationModule):
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
        values: Optional[torch.Tensor] = None,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        representation: Any = None,
        exploit_action: Action = None,
    ) -> Action:
        return torch.distributions.Categorical(values).sample().item()
