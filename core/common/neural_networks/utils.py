from typing import List, Optional

import torch
import torch.nn as nn

from torch.func import stack_module_state

ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softmax": nn.Softmax,
}


def mlp_block(
    input_dim: int,
    hidden_dims: Optional[List[int]],
    output_dim: int = 1,
    use_batch_norm: bool = False,
    use_layer_norm: bool = False,
    hidden_activation: str = "relu",
    last_activation: Optional[str] = None,
) -> nn.Module:
    """
    A simple MLP which can be reused to create more complex networks
    Args:
        input_dim: dimension of the input layer
        hidden_dims: a list of dimensions of the hidden layers
        output_dim: dimension of the output layer
        use_batch_norm: whether to use batch_norm or not in the hidden layers
        hidden_activation: activation function used for hidden layers
        last_activation: this is optional, if need activation for layer, set this input
                        otherwise, no activation is applied on last layer
    Returns:
        an nn.Sequential module consisting of mlp layers
    """
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(ACTIVATION_MAP[hidden_activation]())

    layers.append(nn.Linear(dims[-2], dims[-1]))
    if last_activation is not None:
        layers.append(ACTIVATION_MAP[last_activation]())
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


## To do: the name of this function needs to be revised to xavier_init_weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def update_target_network(target_network, source_network, tau):
    # Q_target = (1 - tao) * Q_target + tao*Q
    target_net_state_dict = target_network.state_dict()
    source_net_state_dict = source_network.state_dict()
    for key in source_net_state_dict:
        target_net_state_dict[key] = (
            tau * source_net_state_dict[key] + (1 - tau) * target_net_state_dict[key]
        )

    target_network.load_state_dict(target_net_state_dict)


def ensemble_forward(models: List[nn.Module], features: torch.Tensor) -> torch.Tensor:
    # followed example in https://pytorch.org/docs for ensembling
    batch_size = features.shape[0]
    features = features.permute((1, 0, 2))

    def wrapper(params, buffers, data):
        return torch.func.functional_call(models[0], (params, buffers), data)

    params, buffers = stack_module_state(models)
    values = torch.vmap(wrapper)(params, buffers, features).view(
        (-1, batch_size)
    )  # (ensemble_size, batch_size)

    # change shape to (batch_size, ensemble_size)
    return values.permute(1, 0)


def update_target_networks(list_of_target_networks, list_of_source_networks, tau):
    """
    Args:
        list_of_target_networks: nn.ModuleList() of nn.Module()
        list_of_source_networks: nn.ModuleList() of nn.Module()
        tau: parameter for soft update
    """
    # Q_target = (1 - tao) * Q_target + tao*Q
    for target_network, source_network in zip(
        list_of_target_networks, list_of_source_networks
    ):
        update_target_network(target_network, source_network, tau)
        # target_net_state_dict = target_network.state_dict()
        # source_net_state_dict = source_network.state_dict()
        # for key in source_net_state_dict:
        #     target_net_state_dict[key] = (
        #         tau * source_net_state_dict[key]
        #         + (1 - tau) * target_net_state_dict[key]
        #     )

        # target_network.load_state_dict(target_net_state_dict)
