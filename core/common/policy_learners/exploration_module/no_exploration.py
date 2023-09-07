import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.core.common.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)


class NoExploration(ExplorationModule):
    """
    An exploration module that does not explore.
    """

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action = None,
        values: torch.Tensor = None,
        representation: torch.Tensor = None,
    ) -> Action:
        if exploit_action is not None:
            # TODO clean up to have NoExploration always argmax on values
            return exploit_action
        values = values.view(-1, action_space.n)
        return torch.argmax(values, dim=1).squeeze()
