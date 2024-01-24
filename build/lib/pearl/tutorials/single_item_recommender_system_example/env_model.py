# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors

import torch
import torch.nn as nn


class SequenceClassificationModel(nn.Module):
    def __init__(
        self,
        observation_dim,
        hidden_dim=128,
        state_dim=128,
        num_layers=2,
    ) -> None:
        super(SequenceClassificationModel, self).__init__()
        self.lstm = nn.LSTM(
            num_layers=num_layers,
            input_size=observation_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.register_buffer(
            "default_cell_representation", torch.zeros((num_layers, hidden_dim))
        )
        self.register_buffer(
            "default_hidden_representation", torch.zeros((num_layers, hidden_dim))
        )

    def forward(self, x, action):
        batch_size = x.shape[0]
        h0 = (
            self.default_hidden_representation.unsqueeze(1)
            .repeat(1, batch_size, 1)
            .detach()
        )
        c0 = (
            self.default_cell_representation.unsqueeze(1)
            .repeat(1, batch_size, 1)
            .detach()
        )
        out, (_, _) = self.lstm(x, (h0, c0))
        mlp_input = out[:, -1, :].view((batch_size, -1))
        return torch.sigmoid(self.mlp(torch.cat([mlp_input, action], dim=-1)))
