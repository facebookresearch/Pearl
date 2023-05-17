from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.observation import Observation
from pearl.api.reward import Value
from pearl.contextual_bandits.contextual_bandit_environment import (
    ContextualBanditEnvironment,
)


class RewardIsEqualToTenTimesActionContextualBanditEnvironment(
    ContextualBanditEnvironment
):
    """
    A example implementation of a contextual bandit environment.
    """

    def __init__(self, action_space: ActionSpace):
        self._action_space = action_space

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    def reset(self) -> (Observation, ActionSpace):
        # Function returning the context and the available action space
        # Here, we use no context (None), but we could return varied implementations.
        return None, self.action_space

    def get_reward(self, action: Action) -> Value:
        # Here goes the code for computing the reward given an action on the current state
        # In this example, the reward is 10 times the digit representing the action.
        return action * 10

    def render(self):
        # Either print or open rendering of environment (optional).
        pass

    def close(self):
        # Close resources (files etc)
        pass

    def __str__(self):
        return "Bandit with reward = 10 * action index"
