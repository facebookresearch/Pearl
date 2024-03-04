# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ...utils import create_normal_pdf_training_data


class TestVanillaMlps(unittest.TestCase):
    def setUp(self) -> None:
        self.x_dim = 4
        self.num_data_points = 1000
        self.learning_rate = 0.001
        self.batch_size = 10
        self.num_epochs = 10

        # create a data set of (x, y) and form a dataloader
        x, y = create_normal_pdf_training_data(
            input_dim=self.x_dim, num_data_points=self.num_data_points
        )
        self.train_dataset = TensorDataset(x, y)
        self.train_dl = DataLoader(self.train_dataset, self.batch_size, shuffle=True)

    def test_vanilla_mlps(self) -> None:
        """
        mlps should be able to fit a simple function and the loss value
        should get close to zero
        """

        network = VanillaValueNetwork(
            input_dim=self.x_dim, hidden_dims=[64, 64], output_dim=1, dropout_ratio=0.1
        )
        optimizer = optim.AdamW(
            network.parameters(), lr=self.learning_rate, amsgrad=True
        )
        criterion = torch.nn.MSELoss()
        losses = []

        for _ in range(self.num_epochs):
            # looping over entire dataset
            for x_batch, y_batch in self.train_dl:
                pred = network(x_batch)[:, 0]  # generate predictions
                loss = criterion(pred, y_batch)  # mse loss
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.assertTrue(
            sum(losses[1:10]) > sum(losses[-10:])
        )  # loss should decrease over learning steps
