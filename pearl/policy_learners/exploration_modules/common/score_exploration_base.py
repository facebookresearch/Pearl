from abc import abstractmethod
from typing import Optional
from warnings import warn

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
    ExplorationType,
)
from pearl.utils.functional_utils.learning.action_utils import get_model_actions
from pearl.utils.tensor_like import assert_is_tensor_like


class ScoreExplorationBase(ExplorationModule):
    """
    Value exploration base module.
    Specific exploration modules need to implement `get_scores`.
    Actions with highest scores will be chosen.
    """

    def __init__(self) -> None:
        super().__init__()
        self.exploration_type: ExplorationType = ExplorationType.VALUE

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: Optional[torch.Tensor] = None,
        action_availability_mask: Optional[torch.Tensor] = None,
        exploit_action: Optional[Action] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        """
        Args:
            subjective_state is in shape of (batch_size, feature_size) or (feature_size)
            for a single transition
            values is in shape of (batch_size, action_count) or (action_count)
        Returns:
            return shape(batch_size,)
        """
        if exploit_action is not None:
            warn(
                "exploit_action shouldn't be used. use `values` instead",
                DeprecationWarning,
            )
            return exploit_action
        assert values is not None
        scores = self.get_scores(
            subjective_state=subjective_state,
            action_space=action_space,
            values=values,
            representation=representation,
        )  # shape: (batch_size, action_count)
        scores = assert_is_tensor_like(scores)
        selected_action = get_model_actions(scores, action_availability_mask)
        return selected_action.squeeze()

    @abstractmethod
    def get_scores(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        values: torch.Tensor,
        exploit_action: Optional[Action] = None,
        representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        """
        Get the scores for each action.

        Args:
            subjective_state is in shape of (batch_size, feature_size) or (feature_size)
            for a single transition
            values is in shape of (batch_size, action_count) or (action_count)
        Returns:
            return shape(batch_size, action_count)
        """
        pass
