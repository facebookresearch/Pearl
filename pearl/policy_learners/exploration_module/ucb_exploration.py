from typing import Iterable

import torch

from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_module.value_exploration_base import (
    ValueExplorationBase,
)
from pearl.utils.action_spaces import DiscreteActionSpace


# TODO: Assumes discrete gym action space
class UCBExploration(ValueExplorationBase):
    """
    UCB exploration module.
    """

    def __init__(self) -> None:
        super(UCBExploration, self).__init__()
        self.action_execution_count = {}
        self.action_executed = torch.tensor(1)

    # TODO: We should make discrete action space itself iterable
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: DiscreteActionSpace,
        values: Iterable[float],
        representation: torch.Tensor = None,
    ) -> Action:
        exploration_bonus = torch.zeros(
            (available_action_space.n)
        )  # (action_space_size)
        for action in available_action_space.actions:
            if action not in self.action_execution_count:
                self.action_execution_count[action] = 1
            exploration_bonus[action] = torch.sqrt(
                torch.log(self.action_executed) / self.action_execution_count[action]
            )

        values = torch.tensor(values)  # (action_space_size)
        selected_action = torch.argmax(values + exploration_bonus).item()
        self.action_execution_count[selected_action] += 1
        self.action_executed += 1
        return selected_action
