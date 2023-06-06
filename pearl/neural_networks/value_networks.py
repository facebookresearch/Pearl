"""
This module defines several types of value neural networks.

Constants:
    StateActionValueNetworkType: a type (and therefore a callable) getting state_dim, action_dim, hidden_dims, output_dim and producing a neural network with
    able to evaluate a state-action pair, consisting of the concatenation of feature tensors for each one with the indicated dimensions.
"""


from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pearl.policy_learners.extend_state_feature import (
    extend_state_feature_by_available_action_space,
)
from torch import Tensor


class VanillaValueNetwork(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: Optional[List[int]], output_dim: int = 1
    ) -> None:

        super(VanillaValueNetwork, self).__init__()

        self._dims = [input_dim] + hidden_dims + [output_dim]
        self._layers = []
        for i in range(len(self._dims) - 2):
            self._layers.append(nn.Linear(self._dims[i], self._dims[i + 1]))
            self._layers.append(nn.ReLU())

        self._layers.append(nn.Linear(self._dims[-2], self._dims[-1]))
        self._model = nn.Sequential(*self._layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    # default initialization in linear and conv layers of a nn.Sequential model is Kaiming
    def xavier_init(self) -> None:
        for layer in self._model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def get_batch_action_value(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch=torch.Tensor,
    ):
        x = torch.cat([state_batch, action_batch], dim=1)
        return self.forward(x).view(-1)  # (batch_size)


class VanillaStateActionValueNetwork(VanillaValueNetwork):
    def __init__(self, state_dim, action_dim, hidden_dims, output_dim):
        super().__init__(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )


class DuelingStateActionValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims,
        output_dim,
        value_hidden_dims: Optional[List[int]] = None,
        advantage_hidden_dims: Optional[List[int]] = None,
    ):
        super(DuelingStateActionValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # feature arch
        self.feature_arch = VanillaValueNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=state_dim,
        )

        # value arch
        self.value_arch = VanillaValueNetwork(
            input_dim=state_dim,
            hidden_dims=[state_dim // 2]
            if value_hidden_dims is None
            else value_hidden_dims,
            output_dim=output_dim,  # output_dim=1
        )

        # advantage arch
        self.advantage_arch = VanillaValueNetwork(
            input_dim=state_dim + action_dim,
            hidden_dims=[state_dim // 2]
            if advantage_hidden_dims is None
            else advantage_hidden_dims,
            output_dim=output_dim,  # output_dim=1
        )

    def forward(self, x):
        """
        The input x is the concatenation of features of [state , action] together
        The archtecture is as follows:
        state --> feature_arch -----> value_arch --> value(s)-----------------------\
                                 |                                                   ---> add --> Q(s,a)
        action --------------concat-> advantage_arch --> advantage(s, a)--- -mean --/
        """
        assert x.shape[-1] == self.state_dim + self.action_dim
        state_feature = x[..., 0 : self.state_dim]
        action_feature = x[..., self.state_dim :]

        # feature arch : state --> feature
        processed_state_feature = F.relu(self.feature_arch(state_feature))

        # value arch : feature --> value
        value = self.value_arch(processed_state_feature)

        # advantage arch : [feature, actions] --> advantage
        feature_action = torch.cat((processed_state_feature, action_feature), dim=-1)
        assert feature_action.shape == x.shape
        advantage = self.advantage_arch(feature_action)
        advantage_mean = torch.mean(advantage, dim=-2, keepdim=True)  # -2 is action dim
        advantage = advantage - advantage_mean
        return value + advantage

    def get_batch_action_value(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch=torch.Tensor,
    ):
        """
        In DUELING_DQN, extend input tensor to include available actions before passing to Q network.
        Then collect the q value of the specific action that we are interested in.
        """
        # calculate the q value of all available actions
        state_multi_actions_batch = extend_state_feature_by_available_action_space(
            state_batch=state_batch,
            curr_available_actions_batch=curr_available_actions_batch,
        )
        # collect Q values of the given action
        values_multi_actions = self.forward(
            state_multi_actions_batch
        )  # (batch_size x actions) , values of a single state with multi-actions

        # gather only the q value of the action that we are interested in.
        action_idx = (
            torch.argmax(action_batch, dim=1).unsqueeze(-1).unsqueeze(-1)
        )  # one_hot to decimal
        state_action_values = torch.gather(values_multi_actions, 1, action_idx).view(
            -1
        )  # (batch_size), value of single state with single action of interest
        return state_action_values


StateActionValueNetworkType = Callable[[int, int, List[int], int], nn.Module]
