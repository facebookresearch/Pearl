import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace

from pearl.api.observation import Observation
from pearl.api.reward import Value
from pearl.contextual_bandits.contextual_bandit_environment import (
    ContextualBanditEnvironment,
)


class ContextualBanditLinearEnvironment(ContextualBanditEnvironment):
    """
    A Contextual Bandit Environment where the reward is linearly mapped from the context feature representation.

    Following

    Lihong Li, Wei Chu, John Langford, Robert E. Schapire (2010), "A Contextual-Bandit Approach to Personalized News Article Recommendation,"

    the observation provided by the environment is a tensor concatenating the *context* for each arm of the bandit.
    The context for an arm is, at a minimum, a feature vector for the arm, but may be prepended with extra contextual information shared by all arms
    (for example, a user for whom a news article is being recommended).
    This implementation currently only provides the arms feature vectors in the observation.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        context_dim: int = 4,
        reward_noise_sigma: float = 0.0,
        context_noise_sigma: float = 0.0,
        simple_linear_mapping: bool = False,
    ):
        """
        Args:
            action_space (ActionSpace): the environment's action space
            context_dim (int): the number of dimensions in the feature representation of contexts
            reward_noise_sigma (float): the standard deviation of the noise added to the reward
            context_noise_sigma (float): the standard deviation of the noise added to the context
            simple_linear_mapping (bool): if True, reward is simply the sum of the arm features (debugging purposes)
        """
        self._action_space = action_space
        self._context_dim = context_dim
        self.reward_noise_sigma = reward_noise_sigma
        self.context_noise_sigma = context_noise_sigma
        self._simple_linear_mapping = simple_linear_mapping

        self._features_of_all_arms = self._generate_features_of_all_arms()
        self._linear_mapping = self.make_initial_linear_mapping()

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def context_dim(self) -> int:
        return self._context_dim

    @property
    def features_of_all_arms(self) -> torch.Tensor:
        return self._features_of_all_arms

    def _generate_features_of_all_arms(self):
        features_of_all_arms = torch.rand(
            self.action_space.n, self.context_dim
        )  # features of each arm. (num of action, num of features)
        return features_of_all_arms

    @property
    def linear_mapping(self) -> torch.nn.Module:
        return self._linear_mapping

    def make_initial_linear_mapping(
        self,
    ) -> torch.nn.Module:
        """
        The function that maps context to reward (always linear).
        The input dimension (in_features) is Environment context_dim.
        The output (reward) is a scalar, so outout dimension (out_features) is 1.
        """
        return torch.nn.Linear(in_features=self.context_dim, out_features=1)

    def reset(self) -> (Observation, ActionSpace):
        """
        Provides the observation and action space to the agent.
        """
        features_of_available_arms = (
            self.features_of_all_arms
        )  # TODO [optional]: apply mask for available arms
        observation = features_of_available_arms
        if self.context_noise_sigma > 0.0:
            observation += torch.rand_like(observation) * self.context_noise_sigma
        return observation, self.action_space

    def get_reward(self, action: Action) -> Value:
        """
        Given action, environment will return the reward associated of this action
        """
        context = self.get_context_for_arm(
            action=action  # action is index in action_space
        )  # (num of actions * num of features)
        reward = self._compute_reward_from_context(context)
        return reward

    def get_context_for_arm(self, action: int) -> torch.Tensor:
        assert action in range(self._action_space.n)  # action is index in action_space
        return self.features_of_all_arms[action]

    def _compute_reward_from_context(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if self._simple_linear_mapping:
            reward = torch.sum(context).unsqueeze(
                dim=0
            )  # assume the linear relationship between context and reward : r_k = ONES @ g(f_k). This is convenient for debugging algorithms when the algorithms are being developed.
        else:
            reward = self._compute_reward_from_context_using_linear_mapping(context)
        return reward

    def _compute_reward_from_context_using_linear_mapping(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        We assume there is linear relationship between context and reward : r_k = g(f_k)
        The g() is a linear function mapping context to reward
        The output is reward, a Value.
        """
        # map features to rewards through a linear function W

        reward = self.linear_mapping(context)

        if self.reward_noise_sigma > 0.0:
            # # add some Gaussian noise to each reward
            noise = torch.randn_like(reward) * self.reward_noise_sigma
            noisy_reward = reward + noise
            return noisy_reward
        else:
            return reward

    def render(self):
        # Either print or open rendering of environment (optional).
        pass

    def close(self):
        # Close resources (files etc)
        pass

    def __str__(self):
        return "Bandit with reward which is linearly mapped from arm's (action) feature vector (context)"
