import gym
from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation


class GymEnvironment(Environment):
    """
    A wrapper for `gym.Env` (Gym 0.21) to behave like Pearl's `Environment`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the wrapper to be around a `gym.Env`.
        Args:
            args: positional arguments to be passed on to Gym's environment.
            kwargs: keyword arguments to be passed on to Gym's environment.
        """
        self.env = gym.make(*args, **kwargs)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self) -> (Observation, ActionSpace):
        # Gym version 0.21 still using 'reset' with no arguments
        observation = self.env.reset()
        return observation, self.action_space

    def step(self, action: Action) -> ActionResult:
        # Gym version 0.21 still using 'done' as opposed to 'terminated' and 'truncated'
        observation, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return ActionResult(observation, reward, terminated, truncated, info)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self) -> str:
        return self.env.spec.id
