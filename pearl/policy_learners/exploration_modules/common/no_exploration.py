from typing import Optional

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    SubjectiveState,
)
from pearl.policy_learners.exploration_modules.common.score_exploration_base import (
    ScoreExplorationBase,
)


class NoExploration(ScoreExplorationBase):
    """
    An exploration module that does not explore.
    """

    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: Optional[torch.Tensor] = None,
        exploit_action: Optional[Action] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        if exploit_action is not None:
            raise ValueError("exploit_action shouldn't be used. use `values` instead")
        # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
        return values.view(-1, action_space.n)  # batch_size, action_count
