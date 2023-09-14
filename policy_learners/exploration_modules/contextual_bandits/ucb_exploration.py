from abc import abstractmethod
from typing import Any, final

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.common.value_exploration_base import (
    ValueExplorationBase,
)
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace


# TODO: Assumes discrete gym action space
class UCBExplorationBase(ValueExplorationBase):
    """
    UCB exploration module.
    """

    def __init__(self, alpha: float) -> None:
        super(UCBExplorationBase, self).__init__()
        self._alpha = alpha

    @abstractmethod
    def sigma(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        representation: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state is in shape of (batch_size, feature_size)
        Returns:
            return shape(batch_size, action_count)
        """
        pass

    @final
    def get_ucb_scores(
        self,
        subjective_state: SubjectiveState,
        values: torch.Tensor,
        available_action_space: DiscreteActionSpace,
        representation: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            subjective_state is in shape of (batch_size, feature_size)
            values is in shape of (batch_size, action_count)
        Returns:
            return shape(batch_size, action_count)
        or
        Args:
            subjective_state is in shape of (feature_size)
            values is in shape of (action_count)
        Returns:
            return shape(action_count)
        """
        action_count = available_action_space.n
        values = values.view(-1, action_count)  # (batch_size, action_count)
        sigma = self.sigma(
            subjective_state=subjective_state,
            available_action_space=available_action_space,
            representation=representation,
        )
        # a safe check before reshape sigma into values
        assert sigma.numel() == values.numel()
        sigma = sigma.view(values.shape)
        ucb_scores = values + self._alpha * sigma
        return ucb_scores.squeeze()

    # TODO: We should make discrete action space itself iterable
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        values: torch.Tensor,
        representation: Any = None,
        exploit_action: Action = None,
    ) -> Action:
        """
        Args:
            subjective_state is in shape of (batch_size, feature_size) or (feature_size)
            for a single transition
            values is in shape of (batch_size, action_count) or (action_count)
        Returns:
            return shape(batch_size,)
        """
        ucb = self.get_ucb_scores(
            subjective_state=subjective_state,
            available_action_space=action_space,
            values=values,
            representation=representation,
        )
        ucb = ucb.view(-1, action_space.n)  # batch_size, action_count
        selected_action = torch.argmax(ucb, dim=1)
        return selected_action.squeeze()


class VanillaUCBExploration(UCBExplorationBase):
    """
    Vanilla UCB exploration module with counter.
    """

    def __init__(self) -> None:
        super(UCBExplorationBase, self).__init__(alpha=1)
        self.action_execution_count = {}
        self.action_executed = torch.tensor(1)

    def sigma(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        representation: torch.Tensor = None,
    ) -> torch.Tensor:
        exploration_bonus = torch.zeros(
            (available_action_space.n)
        )  # (action_space_size)
        for action in available_action_space.actions:
            if action not in self.action_execution_count:
                self.action_execution_count[action] = 1
            exploration_bonus[action] = torch.sqrt(
                torch.log(self.action_executed) / self.action_execution_count[action]
            )
        return exploration_bonus

    # TODO: We should make discrete action space itself iterable
    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: DiscreteActionSpace,
        values: torch.Tensor,
        representation: Any = None,
        exploit_action: Action = None,
    ) -> Action:
        selected_action = super().act(
            subjective_state,
            action_space,
            values,
            representation,
            exploit_action,
        )
        self.action_execution_count[selected_action] += 1
        self.action_executed += 1
        return selected_action
