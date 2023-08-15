"""
This module defines several types of value neural networks.

Constants:
    StateActionValueNetworkType: a type (and therefore a callable) getting state_dim, action_dim, hidden_dims, output_dim and producing a neural network with
    able to evaluate a state-action pair, consisting of the concatenation of feature tensors for each one with the indicated dimensions.
"""


from typing import Callable, List, Optional

import torch
import torch.nn as nn
from pearl.core.common.neural_networks.auto_device_nn_module import AutoDeviceNNModule
from pearl.utils.extend_state_feature import (
    extend_state_feature_by_available_action_space,
)
from torch import Tensor


def mlp_block(
    input_dim: int,
    hidden_dims: Optional[List[int]],
    output_dim: int = 1,
    use_batch_norm: bool = False,
) -> nn.Module:
    """
    A simple MLP which can be reused to create more complex networks
    Args:
        input_dim: dimension of the input layer
        hidden_dims: a list of dimensions of the hidden layers
        output_dim: dimension of the output layer
        use_batch_norm: whether to use batch_norm or not in the hidden layers
    Returns:
        an nn.Sequential module consisting of mlp layers
    """
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def conv_block(
    input_channels_count: int,
    output_channels_list: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    paddings: List[int],
    use_batch_norm=False,
) -> nn.Module:
    """
    Reminder: torch.Conv2d layers expect inputs as (batch_size, in_channels, height, width)
    Notes: layer norm is typically not used with CNNs

    Args:
        input_channels_count: number of input channels
        output_channels_list: a list of number of output channels for each convolutional layer
        kernel_sizes: a list of kernel sizes for each layer
        strides: a list of strides for each layer
        paddings: a list of paddings for each layer
        use_batch_norm: whether to use batch_norm or not in the convolutional layers
    Returns:
        an nn.Sequential module consisting of convolutional layers
    """
    layers = []
    for out_channels, kernel_size, stride, padding in zip(
        output_channels_list, kernel_sizes, strides, paddings
    ):
        conv_layer = nn.Conv2d(
            input_channels_count,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        layers.append(conv_layer)
        if use_batch_norm and input_channels_count > 1:
            layers.append(
                nn.BatchNorm2d(input_channels_count)
            )  # input to Batchnorm 2d is the number of input channels
        layers.append(nn.ReLU())
        input_channels_count = out_channels  # number of input channels to next layer is number of output channels of previous layer

    return nn.Sequential(*layers)


class VanillaValueNetwork(AutoDeviceNNModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]],
        output_dim: int = 1,
    ) -> None:

        super(VanillaValueNetwork, self).__init__()
        self._model = mlp_block(input_dim, hidden_dims, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    # default initialization in linear and conv layers of a nn.Sequential model is Kaiming
    def xavier_init(self) -> None:
        for layer in self._model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)


class VanillaCNN(AutoDeviceNNModule):
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
    ):

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
            self._hidden_dims_fully_connected = []
        else:
            self._hidden_dims_fully_connected = hidden_dims_fully_connected

        self._use_batch_norm_conv = use_batch_norm_conv
        self._use_batch_norm_fully_connected = use_batch_norm_fully_connected
        self._output_dim = output_dim

        self._model_cnn = conv_block(
            input_channels_count=self._input_channels,
            output_channels_list=self._output_channels,
            kernel_sizes=self._kernel_sizes,
            strides=self._strides,
            paddings=self._paddings,
            use_batch_norm=self._use_batch_norm_conv,
        )

        self._mlp_input_dims = self.compute_output_dim_model_cnn()
        self._model_fc = mlp_block(
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


class VanillaStateActionValueNetwork(VanillaValueNetwork):
    def __init__(self, state_dim, action_dim, hidden_dims, output_dim):
        super().__init__(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )

    def get_batch_action_value(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch: Optional[torch.Tensor] = None,
    ):
        """
        Args
            batch of state: (batch_size, state_dim)
            batch of action: (batch_size, action_dim)
            curr_available_actions_batch is set to None for vanilla state action value network; and is only used
            in the Duelling architecture (see there for more details)
        Return
            q values of (state, action) pairs: (batch_size)
        """

        x = torch.cat([state_batch, action_batch], dim=-1)
        return self.forward(x).view(-1)


class DuelingStateActionValueNetwork(AutoDeviceNNModule):
    """
    Dueling architecture contains state arch, value arch, and advantage arch.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims,
        output_dim,
        value_hidden_dims: Optional[List[int]] = None,
        advantage_hidden_dims: Optional[List[int]] = None,
        state_hidden_dims: Optional[List[int]] = None,
    ):
        super(DuelingStateActionValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # state arch
        self.state_arch = VanillaValueNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims if state_hidden_dims is None else state_hidden_dims,
            output_dim=hidden_dims[-1],
        )

        # value arch
        self.value_arch = VanillaValueNetwork(
            input_dim=hidden_dims[-1],  # same as state_arch output dim
            hidden_dims=hidden_dims if value_hidden_dims is None else value_hidden_dims,
            output_dim=output_dim,  # output_dim=1
        )

        # advantage arch
        self.advantage_arch = VanillaValueNetwork(
            input_dim=hidden_dims[-1] + action_dim,  # state_arch out dim + action_dim
            hidden_dims=hidden_dims
            if advantage_hidden_dims is None
            else advantage_hidden_dims,
            output_dim=output_dim,  # output_dim=1
        )

    def forward(self, state, action):
        """
        Args:
            batch of state: (batch_size, state_dim), batch of action: (batch_size, action_dim)
        Returns:
            batch of Q(s,a): (batch_size)

        The archtecture is as follows:
        state --> state_arch -----> value_arch --> value(s)-----------------------\
                                 |                                                   ---> add --> Q(s,a)
        action --------------concat-> advantage_arch --> advantage(s, a)--- -mean --/
        """
        assert state.shape[-1] == self.state_dim
        assert action.shape[-1] == self.action_dim

        # state feature arch : state --> feature
        state_features = self.state_arch(
            state
        )  # shape: (?, state_dim); state_dim is the output dimension of state_arch mlp

        # value arch : feature --> value
        state_value = self.value_arch(state_features)  # shape: (batch_size)

        # advantage arch : [state feature, actions] --> advantage
        state_action_features = torch.cat(
            (state_features, action), dim=-1
        )  # shape: (?, state_dim + action_dim)

        advantage = self.advantage_arch(state_action_features)
        advantage_mean = torch.mean(
            advantage, dim=-2, keepdim=True
        )  # -2 is dimension denoting number of actions
        return state_value + advantage - advantage_mean

    def get_batch_action_value(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch: Optional[torch.Tensor] = None,
    ):
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

            b) when curr_available_actions_batch is not None, extend the state_batch tensor to include available actions
               so, state_batch: (batch_size, state_dim) --> (batch_size, available_action_space_size, state_dim)
               then, do a forward pass from Q network to calculate q-values for (state, all available actions)
               then, choose q values for given (state, action) pair in the batch

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
One can make VanillaValueNetwork to be a special case of TwoTowerStateActionValueNetwork by initializing
linear layers to be an identity map and stopping gradients. This however would be too complex.
"""


class TwoTowerNetwork(AutoDeviceNNModule):
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

    """ This is a horibble way to write this but I will leave it for refactoring which I plan to do next """

    def forward(self, state_action: torch.Tensor):
        state = state_action[..., : self._state_input_dim]
        action = state_action[..., self._state_input_dim :]
        output = self.get_batch_action_value(state_batch=state, action_batch=action)
        return output

    def get_batch_action_value(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        curr_available_actions_batch: Optional[torch.Tensor] = None,
    ):
        state_batch_features = self._state_features.forward(state_batch)
        """ this might need to be done in tensor_based_replay_buffer """
        action_batch_features = self._action_features.forward(
            action_batch.to(torch.float32)
        )
        x = torch.cat([state_batch_features, action_batch_features], dim=-1)
        return self._interaction_features.forward(x).view(-1)  # (batch_size)


"""
With the same initialization parameters as the VanillaStateActionValue Network, i.e. without
specifying the state_output_dims and/or action_outout_dims, we still add a linear layer to
extract state and/or action features.
"""


class TwoTowerStateActionValueNetwork(TwoTowerNetwork):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims,
        output_dim=1,
        state_output_dim=None,
        action_output_dim=None,
        state_hidden_dims=None,
        action_hidden_dims=None,
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


StateActionValueNetworkType = Callable[[int, int, List[int], int], nn.Module]
