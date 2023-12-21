from typing import List, Optional, Tuple, Union

import pandas as pd
import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace

from pearl.api.observation import Observation
from pearl.api.reward import Value
from pearl.utils.instantiations.environments.contextual_bandit_environment import (
    ContextualBanditEnvironment,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class SLCBEnvironment(ContextualBanditEnvironment):
    def __init__(
        self,
        path_filename: str,
        reward_noise_sigma: float = 0.1,
        multi_label: bool = False,
        tr_ts_ratio: float = 1,
        action_embeddings: str = "discrete",
        delim_whitespace: bool = False,
        target_column: int = 0,
        ind_to_drop: Optional[List[int]] = None,
    ) -> None:

        if ind_to_drop is None:
            ind_to_drop = []

        # Load dataset
        with open(path_filename, "rb") as file:
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

        # train_set, _ = torch.utils.data.random_split(tensor, [1, 0])
        train_set = torch.utils.data.TensorDataset(
            tensor[:, 1:], tensor[:, 0]
        )  # featers, targets
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
        self, i: int, action_embeddings: str = "discrete"
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

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, ActionSpace]:
        """
        Provides the observation and action space to the agent.
        """
        data_point = next(iter(self.dataloader_tr))
        label, observation = data_point[1].to(self.device), data_point[0].to(
            self.device
        )
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

    def __str__(self) -> str:
        return "Contextual bandits with CB datasets"


def decimalToBinary(n: int, bits_num: int = 5) -> List[float]:
    string = list("{0:b}".format(int(n)))
    vec = [0.0 for i in range(bits_num)]
    i = bits_num - 1
    for lett in reversed(string):
        vec[i] = float(lett)
        i -= 1
    return vec
