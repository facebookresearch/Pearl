# pyre-ignore-all-errors

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print("gymnasium module not found.")


class GymAvgTorqueWrapper(gym.Wrapper):
    r"""Sparse Reward wrapper for the Pendulum environment."""

    def __init__(self, env):
        super(GymAvgTorqueWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # assumes action is tensor
        cost = (action**2).mean()
        info["cost"] = cost
        return obs, reward, done, truncated, info
