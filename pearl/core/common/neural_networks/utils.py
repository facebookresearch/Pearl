import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def update_target_network(target_network, source_network, tau):
    # Q_target = tao * Q_target + (1-tao)*Q
    target_net_state_dict = target_network.state_dict()
    source_net_state_dict = source_network.state_dict()
    for key in source_net_state_dict:
        target_net_state_dict[key] = (
            tau * source_net_state_dict[key] + (1 - tau) * target_net_state_dict[key]
        )

    target_network.load_state_dict(target_net_state_dict)
