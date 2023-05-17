from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.utils.action_spaces import DiscreteActionSpace


class FixedNumberOfStepsEnvironment(Environment):
    def __init__(self, number_of_steps=100):
        self.number_of_steps_so_far = 0
        self.number_of_steps = number_of_steps
        self._action_space = DiscreteActionSpace([True, False])

    def step(self, action: Action) -> Observation:
        self.number_of_steps_so_far += 1
        return ActionResult(
            observation=self.number_of_steps_so_far,
            reward=self.number_of_steps_so_far,
            terminated=True,
            truncated=True,
            info={},
        )

    def render(self):
        print(self.number_of_steps_so_far)

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    def reset(self):
        return self.number_of_steps_so_far, self.action_space

    def __str__(self):
        return type(self).__name__
