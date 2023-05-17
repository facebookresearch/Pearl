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
    A Contextual Bandit Environment where the reward linear related to arm feature.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        feature_dim: int = 4,
        reward_noise_sigma: float = 0.0,
        simple_linear_relation: bool = False,
    ):
        self._action_space = action_space
        self._feature_dim = feature_dim
        self.reward_noise_sigma = reward_noise_sigma
        self._features_of_all_arms = self.get_action_features()
        self.get_mapping_function()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def features_of_all_arms(self) -> torch.Tensor:
        return self._features_of_all_arms

    @property
    def linear_relation(self, input_dim) -> torch.nn.Module:
        # input is arm feature, output is the reward of this arm
        output_dim = 1  # one arm has one reward
        return torch.nn.Linear(input_dim, output_dim)

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def linear_mapping(self) -> torch.nn.Module:
        return self._linear_mapping

    def get_action_features(self):
        features_of_all_arms = torch.rand(
            self.action_space.n, self.feature_dim
        )  # features of each arm. (num of action, num of features)
        return features_of_all_arms

    def get_mapping_function(
        self,
    ) -> torch.nn.Module:
        """
        The function that maps feature to reward and it is linear function.
        The input dim (in_features) is Environment feature_dim.
        The output (reward) is a scalar, so out dim (out_features) is 1.
        """
        self._linear_mapping = torch.nn.Linear(
            in_features=self.feature_dim, out_features=1
        )

    def get_feature_of_action(self, action: int) -> torch.Tensor:
        assert action in range(self._action_space.n)  # action is index in action_space
        return self.features_of_all_arms[action]

    def reset(self) -> (Observation, ActionSpace):
        """
        Env gives the initial observation (and only observation).
        Env gives obs only at the beginning of the episode with this reset method.

        The agent will take this observation and return an action.
        Env will give reward based on this action.
        Then the CB episode is done.
        """
        features_of_available_arms = (
            self.features_of_all_arms
        )  # TODO [optional]: apply mask for available arms
        obs = features_of_available_arms
        return obs, self.action_space

    def get_reward(self, action: Action) -> Value:
        """
        Given action, Env will return the reward associated of this action
        """
        feature = self.get_feature_of_action(
            action=action  # action is index in action_space
        )  # (num of actions * num of features)
        env_reward = self.feature_to_reward(feature=feature)
        return env_reward  # float

    def feature_to_reward(
        self,
        feature: torch.Tensor,
        simple_linear_relation: bool = False,
    ) -> torch.Tensor:
        if simple_linear_relation:
            reward = torch.sum(feature).unsqueeze(
                dim=0
            )  # assume the linear relationship between feature and reward : r_k = ONES @ g(f_k). This is convenient for debugging algorithms when the algorithms are being developed.
        else:
            reward = self.linear_mapping_feature_to_reward(feature)
        return reward

    def linear_mapping_feature_to_reward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        We assume there is linear relationship between feature and reward : r_k = g(f_k)
        The g() is a linear function mapping feature to reward
        The output is reward, a float number
        """
        # map features to rewards through a linear function W

        mapping = self.linear_mapping
        reward = mapping(feature)

        if self.reward_noise_sigma > 0.0:
            # # add some Gaussian noise to each reward
            noise = torch.randn_like(reward) * self.reward_noise_sigma
            reward_noised = reward + noise
            return reward_noised
        else:
            return reward

    def render(self):
        # Either print or open rendering of environment (optional).
        pass

    def close(self):
        # Close resources (files etc)
        pass

    def __str__(self):
        return "Bandit with reward which is linear mapped from associated arm (action) feature"
