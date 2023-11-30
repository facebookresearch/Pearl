from typing import List, Tuple, Union

import pandas as pd
import torch

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace

from pearl.api.observation import Observation
from pearl.api.reward import Value
from pearl.utils.instantiations.environments.contextual_bandit_environment import (
    ContextualBanditEnvironment,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class ContextualBanditLinearSyntheticEnvironment(ContextualBanditEnvironment):
    """
    A Contextual Bandit synthetic environment where the reward is linearly mapped from the
    context feature representation.
    The purpose of this environment is to simulate the behavior of a Contextual Bandit
    where rewards are modeled linearly from the context features.

    Following

    Lihong Li, Wei Chu, John Langford, Robert E. Schapire (2010),
    "A Contextual-Bandit Approach to Personalized News Article Recommendation,"

    The context for an arm is the concatenation of the observation feature vector
    and the arm feature vevctor.
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
            arm_feature_vector_dim (int): the number of dimensions in the feature representation
                                          of arms
            reward_noise_sigma (float): the standard deviation of the noise added to the reward
            simple_linear_mapping (bool): if True, reward is simply the sum of the arm features
                                          (debugging purposes)
        """
        assert isinstance(action_space, DiscreteActionSpace)
        self._action_space: DiscreteActionSpace = action_space
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
            self._action_space.n,
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
        # TODO: This assumes the action is an int tensor.
        context = self._get_context_for_arm(int(action.item()))
        reward = self._compute_reward_from_context(context)
        return reward

    def get_regret(self, action: Action) -> Value:
        """
        Given action, environment will return regret for choosing this action
        regret == max(reward over all action) - reward for current action
        """
        rewards = [
            self._compute_reward_from_context(self._get_context_for_arm(i))
            for i in range(self._action_space.n)
        ]
        # pyre-fixme[6]: For 1st argument expected
        #  `Iterable[Variable[SupportsRichComparisonT (bound to
        #  Union[SupportsDunderGT[typing.Any], SupportsDunderLT[typing.Any]])]]` but
        #  got `List[Tensor]`.
        return max(rewards) - rewards[action]

    def _get_context_for_arm(self, action: int) -> torch.Tensor:
        assert action in range(self._action_space.n)  # action is index in action_space
        # pyre-fixme[16]: `ContextualBanditLinearSyntheticEnvironment` has no
        #  attribute `_observation`.
        return torch.cat([self._observation, self.features_of_all_arms[action]])

    def _compute_reward_from_context(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if self._simple_linear_mapping:
            reward = torch.sum(context).unsqueeze(dim=0)
            # assume the linear relationship between context and reward :
            # r_k = ONES @ g(f_k). This is convenient for debugging algorithms
            # when the algorithms are being developed.
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


class SLCBEnvironment(ContextualBanditEnvironment):
    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        manifold_path_filename: str,
        reward_noise_sigma: float = 0.1,
        multi_label: bool = False,
        tr_ts_ratio: float = 1,
        action_embeddings: str = "binary_embedding",
        delim_whitespace: bool = False,
        target_column: int = 0,
        ind_to_drop: List[int] = [],
    ):

        #### Load dataset from Manifold
        pathmgr = PathManager()
        pathmgr.register_handler(
            ManifoldPathHandler(timeout_sec=180), allow_override=True
        )
        with pathmgr.open(manifold_path_filename, "rb") as file:
            df = pd.read_csv(file, delim_whitespace=delim_whitespace, header=None)
        # Pre-processing pandas table

        # drop specified columns
        if len(ind_to_drop) > 0:
            df = df.drop(columns=ind_to_drop)
        # place the target at the 0th column
        if target_column != 0:
            name_target = df.columns[target_column]
            name_first_col = df.columns[0]
            df[[name_target, name_first_col]] = df[[name_first_col, name_target]]

        assert not multi_label, "multi label not supported"
        # Cast y labels to numerical values
        df.iloc[:, 0] = df.iloc[:, 0].astype("category")
        df.iloc[:, 0] = df.iloc[:, 0].cat.codes
        unique_labels = pd.unique(df.iloc[:, 0])

        # Cast to torch
        tensor = torch.tensor(df.values)
        tensor = tensor.to(torch.float32)

        # normalize dataset besides of the labels
        means = tensor[:, 1:].mean(dim=0, keepdim=True)
        stds = tensor[:, 1:].std(dim=0, keepdim=True)
        normalized_data = (tensor[:, 1:] - means) / stds
        tensor[:, 1:] = normalized_data

        # pyre-fixme[6]: For 1st argument expected `Dataset[Variable[T]]` but got
        #  `Tensor`
        train_set, _ = torch.utils.data.random_split(tensor, [1, 0])
        dataloader_tr = torch.utils.data.DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
        )

        self.dataloader_tr: torch.utils.data.DataLoader = dataloader_tr

        # Set action space representation of environment
        unique_labels_num = len(unique_labels)
        bits_num = int(torch.ceil(torch.log2(torch.tensor(unique_labels_num))).item())
        self.unique_labels_num: int = unique_labels_num
        self.bits_num: int = bits_num
        self._action_space: ActionSpace = DiscreteActionSpace(
            [self.action_transfomer(i) for i in range(self.unique_labels_num)]
        )
        self._action_dim_env: int = self._action_space[0].shape[0]

        # Set observation dimension
        self.observation_dim: int = tensor.size()[1] - 1  # 0th index is the target

        # Set noise to be added to reward
        self.reward_noise_sigma = reward_noise_sigma
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._current_label: Union[int, None] = None
        self._observation: Union[torch.Tensor, None] = None
        self.bits_num: int = bits_num

    def action_transfomer(
        self, i: int, action_embeddings: str = "binary_embedding"
    ) -> torch.Tensor:
        """
        Transforms the integer action into a tensor.
        i: index of action
        action_embeddings: how to represent actions
        """

        action = None
        if action_embeddings == "discrete":
            return torch.tensor([i])
        elif action_embeddings == "one_hot":
            action = [0.0 for i in range(self.unique_labels_num)]
            action[i] = 1.0
            return torch.tensor(action)
        elif action_embeddings == "binary_embedding":
            action = decimalToBinary(
                i,
                bits_num=self.bits_num,
            )
            return torch.tensor(action)
        else:
            raise Exception("Invalid action_embeddings type")

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    def reset(self) -> Tuple[Observation, ActionSpace]:
        """
        Provides the observation and action space to the agent.
        """
        data_point = next(iter(self.dataloader_tr)).to(self.device)
        label, observation = data_point[:, 0], data_point[:, 1:]
        self._observation = torch.squeeze(observation)
        self._current_label = label.to(int)
        return self._observation, self.action_space

    def get_reward(self, action: Action) -> Value:
        """
        Given action, environment will return the reward associated of this action
        """

        # Set reward to be one if the action matches the label
        reward = torch.tensor(float(action == self._current_label))

        # add Gaussian noise to each reward
        if self.reward_noise_sigma > 0.0:
            noise = torch.randn_like(reward) * self.reward_noise_sigma
            reward += noise

        return reward.item()

    def get_regret(self, action: Action) -> Value:
        """
        Given action, environment will return expected regret for choosing this action
        For supervised learning CB the regret is the indicator whether the action is currect
        """
        return float(action != self._current_label)

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
        return "Contextual bandits with CB datasets"


def decimalToBinary(n: int, bits_num: int = 5) -> List[float]:
    string = list("{0:b}".format(int(n)))
    vec = [0.0 for i in range(bits_num)]
    i = bits_num - 1
    for lett in reversed(string):
        vec[i] = float(lett)
        i -= 1
    return vec
