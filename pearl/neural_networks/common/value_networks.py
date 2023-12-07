# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This module defines several types of value neural networks.
"""


from abc import ABC
from typing import Any, List, Optional

import torch
import torch.nn as nn
from pearl.neural_networks.common.epistemic_neural_networks import Ensemble

from pearl.neural_networks.sequential_decision_making.q_value_network import (
    DistributionalQValueNetwork,
    QValueNetwork,
)
from pearl.utils.functional_utils.learning.extend_state_feature import (
    extend_state_feature_by_available_action_space,
)
from torch import Tensor

from .utils import conv_block, mlp_block


class ValueNetwork(nn.Module, ABC):
    """
    An interface for value neural networks.
    It does not add any required methods to those already present in
    its super classes.
    Its purpose instead is just to serve as an umbrella type for all value networks.
    """


class VanillaValueNetwork(ValueNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int = 1,
        **kwargs: Any,
    ) -> None:
        super(VanillaValueNetwork, self).__init__()
        self._model: nn.Module = mlp_block(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    # default initialization in linear and conv layers of a F.sequential model is Kaiming
    def xavier_init(self) -> None:
        for layer in self._model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)


class VanillaCNN(ValueNetwork):
    """
    Vanilla CNN with a convolutional block followed by an mlp block.
    Args:
        input_width: width of the input
        input_height: height of the input
        input_channels_count: number of input channels
        kernel_sizes: list of kernel sizes for the convolutional layers
        output_channels_list: list of number of output channels for each convolutional layer
        strides: list of strides for each layer
        paddings: list of paddings for each layer
        hidden_dims_fully_connected: a list of dimensions of the hidden layers in the mlp
        use_batch_norm_conv: whether to use batch_norm in the convolutional layers
        use_batch_norm_fully_connected: whether to use batch_norm in the fully connected layers
        output_dim: dimension of the output layer
    Returns:
        An nn.Sequential module consisting of a convolutional block followed by an mlp.
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_channels_count: int,
        kernel_sizes: List[int],
        output_channels_list: List[int],
        strides: List[int],
        paddings: List[int],
        hidden_dims_fully_connected: Optional[
            List[int]
        ] = None,  # hidden dims for fully connected layers
        use_batch_norm_conv: bool = False,
        use_batch_norm_fully_connected: bool = False,
        output_dim: int = 1,  # dimension of the final output
    ) -> None:

        assert (
            len(kernel_sizes)
            == len(output_channels_list)
            == len(strides)
            == len(paddings)
        )
        super(VanillaCNN, self).__init__()

        self._input_channels = input_channels_count
        self._input_height = input_height
        self._input_width = input_width
        self._output_channels = output_channels_list
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._paddings = paddings
        if hidden_dims_fully_connected is None:
            self._hidden_dims_fully_connected: List[int] = []
        else:
            self._hidden_dims_fully_connected: List[int] = hidden_dims_fully_connected

        self._use_batch_norm_conv = use_batch_norm_conv
        self._use_batch_norm_fully_connected = use_batch_norm_fully_connected
        self._output_dim = output_dim

        self._model_cnn: nn.Module = conv_block(
            input_channels_count=self._input_channels,
            output_channels_list=self._output_channels,
            kernel_sizes=self._kernel_sizes,
            strides=self._strides,
            paddings=self._paddings,
            use_batch_norm=self._use_batch_norm_conv,
        )

        self._mlp_input_dims: int = self.compute_output_dim_model_cnn()
        self._model_fc: nn.Module = mlp_block(
            input_dim=self._mlp_input_dims,
            hidden_dims=self._hidden_dims_fully_connected,
            output_dim=self._output_dim,
            use_batch_norm=self._use_batch_norm_fully_connected,
        )

    def compute_output_dim_model_cnn(self) -> int:
        dummy_input = torch.zeros(
            1, self._input_channels, self._input_width, self._input_height
        )
        dummy_output_flattened = torch.flatten(
            self._model_cnn(dummy_input), start_dim=1, end_dim=-1
        )
        return dummy_output_flattened.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        out_cnn = self._model_cnn(x)
        out_flattened = torch.flatten(out_cnn, start_dim=1, end_dim=-1)
        out_fc = self._model_fc(out_flattened)
        return out_fc


class CNNQValueNetwork(VanillaCNN):
    """
    A CNN version of state-action value (Q-value) network.
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_channels_count: int,
        kernel_sizes: List[int],
        output_channels_list: List[int],
        strides: List[int],
        paddings: List[int],
        action_dim: int,
        hidden_dims_fully_connected: Optional[List[int]] = None,
        output_dim: int = 1,
        use_batch_norm_conv: bool = False,
        use_batch_norm_fully_connected: bool = False,
    ) -> None:
        super(CNNQValueNetwork, self).__init__(
            input_width=input_width,
            input_height=input_height,
            input_channels_count=input_channels_count,
            kernel_sizes=kernel_sizes,
            output_channels_list=output_channels_list,
            strides=strides,
            paddings=paddings,
            hidden_dims_fully_connected=hidden_dims_fully_connected,
            use_batch_norm_conv=use_batch_norm_conv,
            use_batch_norm_fully_connected=use_batch_norm_fully_connected,
            output_dim=output_dim,
        )
        # we concatenate actions to state representations in the mlp block of the Q-value network
        self._mlp_input_dims: int = self.compute_output_dim_model_cnn() + action_dim
        self._model_fc: nn.Module = mlp_block(
            input_dim=self._mlp_input_dims,
            hidden_dims=self._hidden_dims_fully_connected,
            output_dim=self._output_dim,
            use_batch_norm=self._use_batch_norm_fully_connected,
        )
        self._action_dim = action_dim

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = state_batch.shape[0]
        state_representation_batch = self._model_cnn(state_batch)
        state_embedding_batch = torch.flatten(
            state_representation_batch, start_dim=1, end_dim=-1
        ).view(batch_size, -1)

        # concatenate actions to state representations and do a forward pass through the mlp_block
        x = torch.cat([state_embedding_batch, action_batch], dim=-1)
        q_values_batch = self._model_fc(x)
        return q_values_batch.view(-1)

    @property
    def action_dim(self) -> int:
        return self._action_dim


class VanillaQValueNetwork(QValueNetwork):
    """
    A vanilla version of state-action value (Q-value) network.
    It leverages the vanilla implementation of value networks by
    using the state-action pair as the input for the value network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_layer_norm: bool = False,
    ) -> None:
        super(VanillaQValueNetwork, self).__init__()
        self._state_dim: int = state_dim
        self._action_dim: int = action_dim
        self._model: nn.Module = mlp_block(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Optional[Tensor] = None,
    ) -> Tensor:
        x = torch.cat([state_batch, action_batch], dim=-1)
        return self.forward(x).view(-1)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class QuantileQValueNetwork(DistributionalQValueNetwork):
    """
    A quantile version of state-action value (Q-value) network.
    For each (state, action) input pairs,
    it returns theta(s,a), the locations of quantiles which parameterize the Q value distribution.

    See the parameterization in QR DQN paper: https://arxiv.org/pdf/1710.10044.pdf for more details.

    Assume N is the number of quantiles.
    1) For this parameterization, the quantiles are fixed (1/N),
       while the quantile locations, theta(s,a), are learned.
    2) The return distribution is represented as: Z(s, a) = (1/N) * sum_{i=1}^N theta_i (s,a),
       where (theta_1(s,a), .. , theta_N(s,a)),
    which represent the quantile locations, are outouts of the QuantileQValueNetwork.

    Args:
        num_quantiles: the number of quantiles N, used to approximate the value distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        num_quantiles: int,
        use_layer_norm: bool = False,
    ) -> None:
        super(QuantileQValueNetwork, self).__init__()

        self._model: nn.Module = mlp_block(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=num_quantiles,
            use_layer_norm=use_layer_norm,
        )

        self._state_dim: int = state_dim
        self._action_dim: int = action_dim
        self._num_quantiles: int = num_quantiles
        self.register_buffer(
            "_quantiles", torch.arange(0, self._num_quantiles + 1) / self._num_quantiles
        )
        self.register_buffer(
            "_quantile_midpoints",
            ((self._quantiles[1:] + self._quantiles[:-1]) / 2)
            .unsqueeze(0)
            .unsqueeze(0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def get_q_value_distribution(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
    ) -> Tensor:

        x = torch.cat([state_batch, action_batch], dim=-1)
        return self.forward(x)

    @property
    def quantiles(self) -> Tensor:
        return self._quantiles

    @property
    def quantile_midpoints(self) -> Tensor:
        return self._quantile_midpoints

    @property
    def num_quantiles(self) -> int:
        return self._num_quantiles

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class DuelingQValueNetwork(QValueNetwork):
    """
    Dueling architecture consists of state architecture, value architecture,
    and advantage architecture.

    The architecture is as follows:
    state --> state_arch -----> value_arch --> value(s)-----------------------\
                                |                                              ---> add --> Q(s,a)
    action ------------concat-> advantage_arch --> advantage(s, a)--- -mean --/
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        value_hidden_dims: Optional[List[int]] = None,
        advantage_hidden_dims: Optional[List[int]] = None,
        state_hidden_dims: Optional[List[int]] = None,
    ) -> None:
        super(DuelingQValueNetwork, self).__init__()
        self._state_dim: int = state_dim
        self._action_dim: int = action_dim

        # state architecture
        self.state_arch = VanillaValueNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims if state_hidden_dims is None else state_hidden_dims,
            output_dim=hidden_dims[-1],
        )

        # value architecture
        self.value_arch = VanillaValueNetwork(
            input_dim=hidden_dims[-1],  # same as state_arch output dim
            hidden_dims=hidden_dims if value_hidden_dims is None else value_hidden_dims,
            output_dim=output_dim,  # output_dim=1
        )

        # advantage architecture
        self.advantage_arch = VanillaValueNetwork(
            input_dim=hidden_dims[-1] + action_dim,  # state_arch out dim + action_dim
            hidden_dims=hidden_dims
            if advantage_hidden_dims is None
            else advantage_hidden_dims,
            output_dim=output_dim,  # output_dim=1
        )

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim

        # state feature architecture : state --> feature
        state_features = self.state_arch(
            state
        )  # shape: (?, state_dim); state_dim is the output dimension of state_arch mlp

        # value architecture : feature --> value
        state_value = self.value_arch(state_features)  # shape: (batch_size)

        # advantage architecture : [state feature, actions] --> advantage
        state_action_features = torch.cat(
            (state_features, action), dim=-1
        )  # shape: (?, state_dim + action_dim)

        advantage = self.advantage_arch(state_action_features)
        advantage_mean = torch.mean(
            advantage, dim=-2, keepdim=True
        )  # -2 is dimension denoting number of actions
        return state_value + advantage - advantage_mean

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            batch of states: (batch_size, state_dim)
            batch of actions: (batch_size, action_dim)
            (Optional) batch of available actions (one set of available actions per state):
                    (batch_size, available_action_space_size, action_dim)

            In DUELING_DQN, logic for use with td learning (deep_td_learning)
            a) when curr_available_actions_batch is None, we do a forward pass from Q network
               in this case, the action batch will be the batch of all available actions
               doing a forward pass with mean subtraction is correct

            b) when curr_available_actions_batch is not None,
               extend the state_batch tensor to include available actions,
               that is, state_batch: (batch_size, state_dim)
               --> (batch_size, available_action_space_size, state_dim)
               then, do a forward pass from Q network to calculate
               q-values for (state, all available actions),
               followed by q values for given (state, action) pair in the batch

        TODO: assumes a gym environment interface with fixed action space, change it with masking
        """

        if curr_available_actions_batch is None:
            return self.forward(state_batch, action_batch).view(-1)
        else:
            # calculate the q value of all available actions
            state_repeated_batch = extend_state_feature_by_available_action_space(
                state_batch=state_batch,
                curr_available_actions_batch=curr_available_actions_batch,
            )  # shape: (batch_size, available_action_space_size, state_dim)

            # collect Q values of a state and all available actions
            values_state_available_actions = self.forward(
                state_repeated_batch, curr_available_actions_batch
            )  # shape: (batch_size, available_action_space_size, action_dim)

            # gather only the q value of the action that we are interested in.
            action_idx = (
                torch.argmax(action_batch, dim=1).unsqueeze(-1).unsqueeze(-1)
            )  # one_hot to decimal

            # q value of (state, action) pair of interest
            state_action_values = torch.gather(
                values_state_available_actions, 1, action_idx
            ).view(
                -1
            )  # shape: (batch_size)
        return state_action_values


"""
One can make VanillaValueNetwork to be a special case of TwoTowerQValueNetwork by initializing
linear layers to be an identity map and stopping gradients. This however would be too complex.
"""


class TwoTowerNetwork(QValueNetwork):
    def __init__(
        self,
        state_input_dim: int,
        action_input_dim: int,
        state_output_dim: int,
        action_output_dim: int,
        state_hidden_dims: Optional[List[int]],
        action_hidden_dims: Optional[List[int]],
        hidden_dims: Optional[List[int]],
        output_dim: int = 1,
    ) -> None:

        super(TwoTowerNetwork, self).__init__()

        """
        Input: batch of state, batch of action. Output: batch of Q-values for (s,a) pairs
        The two tower archtecture is as follows:
        state ----> state_feature
                            | concat ----> Q(s,a)
        action ----> action_feature
        """
        self._state_input_dim = state_input_dim
        self._action_input_dim = action_input_dim
        self._state_features = VanillaValueNetwork(
            input_dim=state_input_dim,
            hidden_dims=state_hidden_dims,
            output_dim=state_output_dim,
        )
        self._state_features.xavier_init()
        self._action_features = VanillaValueNetwork(
            input_dim=action_input_dim,
            hidden_dims=action_hidden_dims,
            output_dim=action_output_dim,
        )
        self._action_features.xavier_init()
        self._interaction_features = VanillaValueNetwork(
            input_dim=state_output_dim + action_output_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        self._interaction_features.xavier_init()

    def forward(self, state_action: Tensor) -> Tensor:
        state = state_action[..., : self._state_input_dim]
        action = state_action[..., self._state_input_dim :]
        output = self.get_q_values(state_batch=state, action_batch=action)
        return output

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Optional[Tensor] = None,
    ) -> Tensor:
        state_batch_features = self._state_features.forward(state_batch)
        """ this might need to be done in tensor_based_replay_buffer """
        action_batch_features = self._action_features.forward(
            action_batch.to(torch.get_default_dtype())
        )

        x = torch.cat([state_batch_features, action_batch_features], dim=-1)
        return self._interaction_features.forward(x).view(-1)  # (batch_size)

    @property
    def state_dim(self) -> int:
        return self._state_input_dim

    @property
    def action_dim(self) -> int:
        return self._action_input_dim


"""
With the same initialization parameters as the VanillaQValue Network, i.e. without
specifying the state_output_dims and/or action_outout_dims, we still add a linear layer to
extract state and/or action features.
"""


class TwoTowerQValueNetwork(TwoTowerNetwork):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        state_output_dim: Optional[int] = None,
        action_output_dim: Optional[int] = None,
        state_hidden_dims: Optional[List[int]] = None,
        action_hidden_dims: Optional[List[int]] = None,
    ) -> None:

        super().__init__(
            state_input_dim=state_dim,
            action_input_dim=action_dim,
            state_output_dim=state_dim
            if state_output_dim is None
            else state_output_dim,
            action_output_dim=action_dim
            if action_output_dim is None
            else action_output_dim,
            state_hidden_dims=[] if state_hidden_dims is None else state_hidden_dims,
            action_hidden_dims=[] if action_hidden_dims is None else action_hidden_dims,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )


class EnsembleQValueNetwork(QValueNetwork):
    r"""A Q-value network that uses the `Ensemble` model."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int,
        ensemble_size: int,
        prior_scale: float = 1.0,
    ) -> None:
        super(EnsembleQValueNetwork, self).__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._model = Ensemble(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            ensemble_size=ensemble_size,
            prior_scale=prior_scale,
        )

    @property
    def ensemble_size(self) -> int:
        return self._model.ensemble_size

    def resample_epistemic_index(self) -> None:
        r"""Resamples the epistemic index of the underlying model."""
        self._model._resample_epistemic_index()

    def forward(
        self, x: Tensor, z: Optional[Tensor] = None, persistent: bool = False
    ) -> Tensor:
        return self._model(x, z=z, persistent=persistent)

    def get_q_values(
        self,
        state_batch: Tensor,
        action_batch: Tensor,
        curr_available_actions_batch: Optional[Tensor] = None,
        z: Optional[Tensor] = None,
        persistent: bool = False,
    ) -> Tensor:
        x = torch.cat([state_batch, action_batch], dim=-1)
        return self.forward(x, z=z, persistent=persistent).view(-1)

    @property
    def state_dim(self) -> int:
        return self._state_input_dim

    @property
    def action_dim(self) -> int:
        return self._action_input_dim
