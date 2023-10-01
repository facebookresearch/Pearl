import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
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
        # pyre-fixme[9]: values has type `Tensor`; used as `None`.
        values: torch.Tensor = None,
        # pyre-fixme[9]: representation has type `Tensor`; used as `None`.
        representation: torch.Tensor = None,
    ) -> Action:
        if exploit_action is not None:
            # TODO clean up to have NoExploration always argmax on values
            return exploit_action
        # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
        values = values.view(-1, action_space.n)
        return torch.argmax(values, dim=1).squeeze()
