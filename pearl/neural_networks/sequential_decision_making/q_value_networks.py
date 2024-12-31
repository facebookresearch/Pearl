# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

"""
This file defines PEARL neural network interafaces
User is free to define their own Q(s, a), but would need to inherit from this interface
"""

from __future__ import annotations

import abc
from typing import List, Optional

import torch
from pearl.neural_networks.common.epistemic_neural_networks import Ensemble
from pearl.neural_networks.common.utils import (
    compute_output_dim_model_cnn,
    conv_block,
    mlp_block,
)
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.utils.functional_utils.learning.extend_state_feature import (
    extend_state_feature_by_available_action_space,
)
from pearl.utils.functional_utils.learning.is_one_hot_tensor import is_one_hot_tensor
from torch import nn, Tensor


class QValueNetwork(abc.ABC, nn.Module):
    """
    An interface for state-action value (Q-value) estimators (typically, neural networks).
    These are value neural networks with a special method
    for computing the Q-value for a state-action pair.
    """

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Returns state dimention"""
        ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Returns action dimention"""
        ...

    @abc.abstractmethod
    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns Q(s, a), given s and a
        Args:
            state_batch (torch.Tensor): a batch of state tensors (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of action tensors (batch_size, action_dim)
            curr_available_actions_batch (torch.Tensor, optional): a batch of currently available
                actions (batch_size, available_action_space_size, action_dim)
        Returns:
            Q-values of (state, action) pairs: (batch_size)
        """
        ...


class DistributionalQValueNetwork(abc.ABC, nn.Module):
    """
    An interface for estimators of state-action value distribution (Q-value distribution).
    These are value neural networks with special method for computing the Q-value distribution
    and the expected Q-values for a state-action pair.
    Examples include Categorical DQN, Quantile DQN, IQN etc.
    """

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Returns state dimention"""
        ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Returns action dimention"""
        ...

    @property
    @abc.abstractmethod
    def num_quantiles(self) -> int:
        """Returns number of particles for approximating the quantile distribution"""

    @property
    @abc.abstractmethod
    def quantiles(self) -> torch.Tensor:
        """Returns quantiles of the approximate value distribution"""

    @property
    @abc.abstractmethod
    def quantile_midpoints(self) -> torch.Tensor:
        """Returns midpoints of the quantiles"""

    @abc.abstractmethod
    def get_q_value_distribution(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Returns Z(s, a), a probability distribution over q values, given s and a.
        Note that under a risk neutral measure, Q(s,a) = E[Z(s, a)].
        Args:
            state_batch (torch.Tensor): a batch of state tensors (batch_size, state_dim)
            action_batch (torch.Tensor): a batch of action tensors (batch_size, action_dim)
        Returns:
            approximation of distribution of Q-values of (state, action) pairs
        """
        ...


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
        hidden_dims: list[int],
        output_dim: int,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
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
        state_batch: Tensor,  # (batch_size, state_dim)
        # (batch_size, number of query actions, action_dim) or (batch_size, action_dim)
        action_batch: Tensor,
        curr_available_actions_batch: Tensor | None = None,
    ) -> Tensor:
        assert len(state_batch.shape) == 2
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        if len(action_batch.shape) == 2:
            extended_action_batch = action_batch.unsqueeze(1)
        else:
            extended_action_batch = action_batch
        state_batch = extend_state_feature_by_available_action_space(
            state_batch, extended_action_batch
        )  # (batch_size, number_of_actions_to_query, state_dim)
        x = torch.cat(
            [state_batch, extended_action_batch], dim=-1
        )  # (batch_size, number_of_actions_to_query, (state_dim + action_dim))
        q_values = self.forward(x).squeeze(
            -1
        )  # (batch_size, number_of_actions_to_query)
        return q_values if len(action_batch.shape) == 3 else q_values.squeeze(-1)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class VanillaQValueMultiHeadNetwork(QValueNetwork):
    """
    A vanilla version of state-action value (Q-value) multi-head network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        output_dim: int,  # action space size
        use_layer_norm: bool = False,
    ) -> None:
        super(VanillaQValueMultiHeadNetwork, self).__init__()
        self._state_dim: int = state_dim
        self._action_dim: int = action_dim
        self._output_dim: int = output_dim
        self._model: nn.Module = mlp_block(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def get_q_values(
        self,
        state_batch: Tensor,  # (batch_size, state_dim)
        # (batch_size, number of query actions, action_dim) or (batch_size, action_dim)
        action_batch: Tensor,
        curr_available_actions_batch: Optional[Tensor] = None,
    ) -> Tensor:
        # action representation is assumed to be one-hot
        assert is_one_hot_tensor(action_batch)
        assert self._output_dim == action_batch.shape[-1]  # num actions = action_dim
        assert len(state_batch.shape) == 2
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        if len(action_batch.shape) == 2:
            extended_action_batch = action_batch.unsqueeze(1)
        else:
            extended_action_batch = action_batch

        # We obtain the values for all actions and filter the queries/available actions
        # by multiplying q_values for all actions by the one-hot action batch.
        # We unsqueeze the last dimension to make the q-value column vectors matrices
        # so they can be multiplied with torch.bmm. Afterwards we squeeze the added dimension back.
        q_values = self.forward(state_batch).unsqueeze(
            -1
        )  # (batch_size x num actions x 1)
        q_values = torch.bmm(
            extended_action_batch,  # shape: (batch_size, number of query actions, num actions)
            q_values,  # (batch_size x num actions x 1)
        )  # (batch_size x number of query actions x 1)
        q_values = q_values.squeeze(-1)  # (batch_size x number of query actions)
        return q_values if len(action_batch.shape) == 3 else q_values.squeeze(-1)

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
        hidden_dims: list[int],
        num_quantiles: int,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

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
        state_batch: Tensor,  # (batch_size x state_dim)
        # (batch_size, number of query actions, action_dim) or (batch_size, action_dim)
        action_batch: Tensor,
    ) -> Tensor:
        assert len(state_batch.shape) == 2
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        if len(action_batch.shape) == 2:
            extended_action_batch = action_batch.unsqueeze(1)
        else:
            extended_action_batch = action_batch

        state_batch = extend_state_feature_by_available_action_space(
            state_batch, extended_action_batch
        )  # (batch_size, number_of_actions_to_query, state_dim)
        x = torch.cat(
            [state_batch, extended_action_batch], dim=-1
        )  # (batch_size, number_of_actions_to_query, (state_dim + action_dim))
        q_values = self.forward(
            x
        )  # (batch_size, number_of_actions_to_query, number_of_quantiles)
        return q_values if len(action_batch.shape) == 3 else q_values.squeeze(-2)

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
        hidden_dims: list[int],
        output_dim: int,
        value_hidden_dims: list[int] | None = None,
        advantage_hidden_dims: list[int] | None = None,
        state_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
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
            hidden_dims=(
                hidden_dims if advantage_hidden_dims is None else advantage_hidden_dims
            ),
            output_dim=output_dim,  # output_dim=1
        )

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def get_q_values(
        self,
        state_batch: Tensor,  # (batch_size, state_dim)
        # (batch_size, number of query actions, action_dim) or (batch_size, action_dim)
        action_batch: Tensor,
        curr_available_actions_batch: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            batch of states: (batch_size, state_dim)
            batch of actions: (batch_size, number of query actions, action_dim)
            or (batch_size, action_dim)
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
        assert len(state_batch.shape) == 2
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        if len(action_batch.shape) == 2:
            extended_action_batch = action_batch.unsqueeze(1)
        else:
            extended_action_batch = action_batch

        # state feature architecture : state --> feature
        state_features = self.state_arch(
            state_batch
        )  # shape: (batch_size, state_feature_dim)
        # state_feature_dim is the output dimension of state_arch mlp

        # value architecture : feature --> value
        state_value = self.value_arch(state_features)  # shape: (batch_size, 1)

        # Create a new tensor by repeating state_features for a certain number of times.
        # The number is the max of the number of query actions and the number of available actions.
        # In this way, we do not need to call extend_state_feature_by_available_action_space twice
        # for extended_action_batch and curr_available_actions_batch respectively.
        number_of_query_actions = extended_action_batch.shape[1]
        if (
            curr_available_actions_batch is None
            or number_of_query_actions > curr_available_actions_batch.shape[1]
        ):
            extended_state_features = extend_state_feature_by_available_action_space(
                state_features, extended_action_batch
            )  # (batch_size, number_of_actions_to_query, state_dim)
        else:
            extended_state_features = extend_state_feature_by_available_action_space(
                state_features, curr_available_actions_batch
            )  # (batch_size, number_of_available_actions, state_dim)

        # advantage architecture : [state feature, actions] --> advantage
        state_action_features = torch.cat(
            (
                extended_state_features[:, :number_of_query_actions, :],
                extended_action_batch,
            ),
            dim=-1,
        )  # shape: (batch_size, number_of_actions_to_query, state_dim + action_dim)
        advantage = self.advantage_arch(state_action_features).squeeze(
            -1
        )  # shape: (batch_size, number of query actions)

        if curr_available_actions_batch is None:
            advantage_mean = torch.mean(
                advantage, dim=-1, keepdim=True
            )  # shape: (batch_size, 1)
        else:
            # advantage architecture : [state feature, actions] --> advantage
            state_action_features = torch.cat(
                (
                    extended_state_features[
                        :, : curr_available_actions_batch.shape[1], :
                    ],
                    curr_available_actions_batch,
                ),
                dim=-1,
            )  # shape: (batch_size, action_space_size, state_dim + action_dim)
            available_actions_advantage = self.advantage_arch(
                state_action_features
            ).squeeze(-1)  # shape: (batch_size, action_space_size)
            advantage_mean = torch.mean(
                available_actions_advantage, dim=-1, keepdim=True
            )  # shape: (batch_size, 1)
        q_values = (
            state_value + advantage - advantage_mean
        )  # shape: (batch_size, number of query actions)

        return q_values if len(action_batch.shape) == 3 else q_values.squeeze(-1)


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
        state_hidden_dims: list[int] | None,
        action_hidden_dims: list[int] | None,
        hidden_dims: list[int] | None,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

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
        state_batch: Tensor,  # (batch_size, state_dim)
        # (batch_size, number of query actions, action_dim) or (batch_size, action_dim)
        action_batch: Tensor,
        curr_available_actions_batch: Tensor | None = None,
    ) -> Tensor:
        assert len(state_batch.shape) == 2
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        if len(action_batch.shape) == 2:
            extended_action_batch = action_batch.unsqueeze(1)
        else:
            extended_action_batch = action_batch

        state_batch = extend_state_feature_by_available_action_space(
            state_batch, extended_action_batch
        )  # (batch_size, number_of_actions_to_query, state_dim)

        state_batch_features = self._state_features.forward(state_batch)
        """ this might need to be done in tensor_based_replay_buffer """
        action_batch_features = self._action_features.forward(
            extended_action_batch.to(torch.get_default_dtype())
        )

        x = torch.cat(
            [state_batch_features, action_batch_features], dim=-1
        )  # (batch_size, number_of_actions_to_query, state_feature_dim + action_feature_dim)
        q_values = self._interaction_features.forward(x).squeeze(
            -1
        )  # (batch_size, number_of_actions_to_query)
        return q_values if len(action_batch.shape) == 3 else q_values.squeeze(-1)

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
        hidden_dims: list[int],
        output_dim: int = 1,
        state_output_dim: int | None = None,
        action_output_dim: int | None = None,
        state_hidden_dims: list[int] | None = None,
        action_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__(
            state_input_dim=state_dim,
            action_input_dim=action_dim,
            state_output_dim=(
                state_dim if state_output_dim is None else state_output_dim
            ),
            action_output_dim=(
                action_dim if action_output_dim is None else action_output_dim
            ),
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
        hidden_dims: list[int] | None,
        output_dim: int,
        ensemble_size: int,
        prior_scale: float = 1.0,
    ) -> None:
        super().__init__()
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

    def forward(self, x: Tensor, z: Tensor, persistent: bool = False) -> Tensor:
        return self._model(x, z=z, persistent=persistent)

    def get_q_values(
        self,
        state_batch: Tensor,  # (batch_size, state_dim)
        # (batch_size, number of query actions, action_dim) or (batch_size, action_dim)
        action_batch: Tensor,
        z: Tensor,
        curr_available_actions_batch: Tensor | None = None,
        persistent: bool = False,
    ) -> Tensor:
        assert len(state_batch.shape) == 2
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        if len(action_batch.shape) == 2:
            extended_action_batch = action_batch.unsqueeze(1)
        else:
            extended_action_batch = action_batch

        state_batch = extend_state_feature_by_available_action_space(
            state_batch, extended_action_batch
        )  # (batch_size, number_of_actions_to_query, state_dim)
        x = torch.cat(
            [state_batch, extended_action_batch], dim=-1
        )  # (batch_size, number_of_actions_to_query, (state_dim + action_dim))
        q_values = self.forward(x, z=z, persistent=persistent).squeeze(
            -1
        )  # (batch_size, number_of_actions_to_query)
        return q_values if len(action_batch.shape) == 3 else q_values.squeeze(-1)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class CNNQValueNetwork(QValueNetwork):
    """
    A CNN version of state-action value (Q-value) network.
    The states are assumed to be tensors (input_channels, input_height, input_width)
    and actions are vectors (action_dim).
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_channels_count: int,
        kernel_sizes: list[int],
        output_channels_list: list[int],
        strides: list[int],
        paddings: list[int],
        action_dim: int,
        hidden_dims_fully_connected: list[int] | None = None,
        output_dim: int = 1,
        use_batch_norm_conv: bool = False,
        use_batch_norm_fully_connected: bool = False,
    ) -> None:
        super().__init__()

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
        # we concatenate actions to state representations in the mlp block of the Q-value network
        self._mlp_input_dims: int = (
            compute_output_dim_model_cnn(
                input_channels=input_channels_count,
                input_width=input_width,
                input_height=input_height,
                model_cnn=self._model_cnn,
            )
            + action_dim
        )
        self._model_fc: nn.Module = mlp_block(
            input_dim=self._mlp_input_dims,
            hidden_dims=self._hidden_dims_fully_connected,
            output_dim=self._output_dim,
            use_batch_norm=self._use_batch_norm_fully_connected,
        )
        self._state_dim: int = input_channels_count * input_height * input_width
        self._action_dim = action_dim

    def get_q_values(
        self,
        state_batch: Tensor,  # shape: (batch_size, input_channels, input_height, input_width)
        # shape: (batch_size, number_of_actions_to_query, action_dim) or (batch_size, action_dim)
        action_batch: Tensor,
        curr_available_actions_batch: Tensor | None = None,
    ) -> Tensor:
        assert len(state_batch.shape) == 4
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        if len(action_batch.shape) == 2:
            extended_action_batch = action_batch.unsqueeze(1)
        else:
            extended_action_batch = action_batch

        batch_size = state_batch.shape[0]
        num_query_actions = extended_action_batch.shape[1]
        state_representation_batch = self._model_cnn(
            state_batch / 255.0
        )  # (batch_size, output_channels[-1], output_height, output_width)
        state_representation_batch = state_representation_batch.view(
            batch_size, -1
        )  # (batch_size, state dim)
        # concatenate actions to state representations and do a forward pass through the mlp_block
        state_representation_batch = torch.repeat_interleave(
            state_representation_batch.unsqueeze(1), num_query_actions, dim=1
        )  # (batch_size, number_of_actions_to_query, state_dim)
        x = torch.cat(
            [state_representation_batch, extended_action_batch], dim=-1
        )  # (batch_size, number_of_actions_to_query, (state_dim + action_dim))
        x = x.view(-1, x.shape[-1])
        q_values = self._model_fc(x).reshape(
            batch_size, num_query_actions
        )  # (batch_size, number_of_actions_to_query)
        return q_values if len(action_batch.shape) == 3 else q_values.squeeze(-1)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim


class CNNQValueMultiHeadNetwork(QValueNetwork):
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
        super(CNNQValueMultiHeadNetwork, self).__init__()

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
        # we concatenate actions to state representations in the mlp block of the Q-value network
        self._mlp_input_dims: int = compute_output_dim_model_cnn(
            input_channels=input_channels_count,
            input_width=input_width,
            input_height=input_height,
            model_cnn=self._model_cnn,
        )
        self._model_fc: nn.Module = mlp_block(
            input_dim=self._mlp_input_dims,
            hidden_dims=self._hidden_dims_fully_connected,
            output_dim=self._output_dim,
            use_batch_norm=self._use_batch_norm_fully_connected,
        )
        self._state_dim: int = input_channels_count * input_height * input_width
        self._action_dim = action_dim

    def get_q_values(
        self,
        state_batch: Tensor,  # shape: (batch_size, input_channels, input_height, input_width)
        action_batch: Tensor,  # shape: (batch_size, number_of_actions_to_query, action_dim) or
        #                               (batch_size, action_dim)
        curr_available_actions_batch: Optional[Tensor] = None,
    ) -> Tensor:
        # action representation is assumed to be one-hot
        assert is_one_hot_tensor(action_batch)
        assert self._output_dim == action_batch.shape[-1]  # action_dim = num actions
        assert len(state_batch.shape) == 4
        assert len(action_batch.shape) == 3 or len(action_batch.shape) == 2
        if len(action_batch.shape) == 2:
            action_batch = action_batch.unsqueeze(1)

        batch_size = state_batch.shape[0]
        state_representation_batch = self._model_cnn(
            state_batch / 255.0
        )  # (batch_size x output_channels[-1] x output_height x output_width)
        state_representation_batch = state_representation_batch.view(
            batch_size, -1
        )  # (batch_size x state dim)
        q_values = self._model_fc(state_representation_batch).unsqueeze(
            -1
        )  # (batch_size x num actions x 1)

        q_values = torch.bmm(
            action_batch,  # shape: (batch_size, number_of_actions_to_query, action_dim)
            q_values,  # (batch_size x num actions x 1)
        )  # (batch_size x number_of_actions_to_query x 1)
        q_values = q_values.squeeze(-1)  # (batch_size x number_of_actions_to_query)
        return q_values if len(action_batch) == 3 else q_values.squeeze(-1)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim
