import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace

from pearl.api.observation import Observation
from pearl.api.reward import Value
from pearl.utils.instantiations.environments.contextual_bandit_environment import (
    ContextualBanditEnvironment,
)


class ContextualBanditLinearSyntheticEnvironment(ContextualBanditEnvironment):
    """
    A Contextual Bandit synthetic environment where the reward is linearly mapped from the context feature representation.
    The purpose of this environment is to simulate the behavior of a Contextual Bandit where rewards are modeled linearly from the context features.

    Following

    Lihong Li, Wei Chu, John Langford, Robert E. Schapire (2010), "A Contextual-Bandit Approach to Personalized News Article Recommendation,"

    the context for an arm is the concatenation of the observation feature vector and the arm feature vevctor.
    """

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        action_space: ActionSpace,
        observation_dim: int = 0,
        arm_feature_vector_dim: int = 4,
        reward_noise_sigma: float = 0.0,
        simple_linear_mapping: bool = False,
    ):
        """
        Args:
            action_space (ActionSpace): the environment's action space
            observation_dim (int): the number of dimensions in the observation feature vector
            arm_feature_vector_dim (int): the number of dimensions in the feature representation of arms
            reward_noise_sigma (float): the standard deviation of the noise added to the reward
            simple_linear_mapping (bool): if True, reward is simply the sum of the arm features (debugging purposes)
        """
        self._action_space = action_space
        self.observation_dim = observation_dim
        self._arm_feature_vector_dim = arm_feature_vector_dim
        self.reward_noise_sigma = reward_noise_sigma
        self._simple_linear_mapping = simple_linear_mapping

        # pyre-fixme[4]: Attribute must be annotated.
        self._features_of_all_arms = self._generate_features_of_all_arms()
        # pyre-fixme[4]: Attribute must be annotated.
        self._linear_mapping = self._make_initial_linear_mapping()

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def arm_feature_vector_dim(self) -> int:
        return self._arm_feature_vector_dim

    @property
    def features_of_all_arms(self) -> torch.Tensor:
        return self._features_of_all_arms

    @property
    def linear_mapping(self) -> torch.nn.Module:
        return self._linear_mapping

    # pyre-fixme[3]: Return type must be annotated.
    def _generate_features_of_all_arms(self):
        features_of_all_arms = torch.rand(
            # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
            self.action_space.n,
            self.arm_feature_vector_dim,
        )  # features of each arm. (num of action, num of features)
        return features_of_all_arms

    def _make_initial_linear_mapping(
        self,
    ) -> torch.nn.Module:
        """
        The function that maps context to reward (always linear).
        The input dimension (in_features) is observation dimension + arm feature vector dimension.
        The output (reward) is a scalar, so output dimension (out_features) is 1.
        """
        return torch.nn.Linear(
            in_features=self.observation_dim + self.arm_feature_vector_dim,
            out_features=1,
        )

    # pyre-fixme[31]: Expression `ActionSpace)` is not a valid type.
    def reset(self) -> (Observation, ActionSpace):
        """
        Provides the observation and action space to the agent.
        """
        # pyre-fixme[16]: `ContextualBanditLinearSyntheticEnvironment` has no
        #  attribute `_observation`.
        self._observation = torch.rand(self.observation_dim)
        return self._observation, self.action_space

    def get_reward(self, action: Action) -> Value:
        """
        Given action, environment will return the reward associated of this action
        """
        context = self._get_context_for_arm(action)
        reward = self._compute_reward_from_context(context)
        # pyre-fixme[7]: Expected `Number` but got `Tensor`.
        return reward

    def get_regret(self, action: Action) -> Value:
        """
        Given action, environment will return regret for choosing this action
        regret == max(reward over all action) - reward for current action
        """
        rewards = [
            self._compute_reward_from_context(self._get_context_for_arm(i))
            # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
            for i in range(self._action_space.n)
        ]
        # pyre-fixme[6]: For 1st argument expected
        #  `Iterable[Variable[SupportsRichComparisonT (bound to
        #  Union[SupportsDunderGT[typing.Any], SupportsDunderLT[typing.Any]])]]` but
        #  got `List[Tensor]`.
        return max(rewards) - rewards[action]

    def _get_context_for_arm(self, action: int) -> torch.Tensor:
        # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
        assert action in range(self._action_space.n)  # action is index in action_space
        # pyre-fixme[16]: `ContextualBanditLinearSyntheticEnvironment` has no
        #  attribute `_observation`.
        return torch.cat([self._observation, self.features_of_all_arms[action]])

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
            # add some Gaussian noise to each reward
            noise = torch.randn_like(reward) * self.reward_noise_sigma
            noisy_reward = reward + noise
            return noisy_reward
        else:
            return reward

    # pyre-fixme[3]: Return type must be annotated.
    def render(self):
        # Either print or open rendering of environment (optional).
        pass

    # pyre-fixme[3]: Return type must be annotated.
    def close(self):
        # Close resources (files etc)
        pass

    # pyre-fixme[3]: Return type must be annotated.
    def __str__(self):
        return "Bandit with reward linearly mapped from context feature vector"
