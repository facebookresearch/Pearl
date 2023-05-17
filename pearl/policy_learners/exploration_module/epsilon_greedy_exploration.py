import random

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)


class EGreedyExploration(ExplorationModule):
    """
    epsilon Greedy exploration module.
    """

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def act(
        self,
        subjective_state: SubjectiveState,
        action_space: ActionSpace,
        exploit_action: Action,
    ) -> Action:
        if random.random() < self.epsilon:
            return action_space.sample()

        return exploit_action
