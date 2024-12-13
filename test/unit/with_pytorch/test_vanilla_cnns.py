# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import numpy as np

import torch
import torchvision
from pearl.neural_networks.common.value_networks import CNNValueNetwork

from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


class TestCNNValueNetworks(unittest.TestCase):
    def setUp(self) -> None:
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_dataset = torchvision.datasets.MNIST(
            root="./", train=True, transform=transform, download=True
        )

        self.num_data_points = 5000
        self.mnist_train_dataset = Subset(
            mnist_dataset,
            # pyre-fixme[6]: For 2nd argument expected `Sequence[int]` but got
            #  `ndarray[typing.Any, dtype[typing.Any]]`.
            np.arange(1, self.num_data_points),
        )
        self.learning_rate = 0.001
        self.batch_size = 64
        self.num_epochs = 10

        self.train_dl = DataLoader(
            self.mnist_train_dataset, self.batch_size, shuffle=True
        )

    def test_cnns(self) -> None:
        """
        a simple cnn should be able to fit the mnist digit dataset and the training
        accuracy should be close to 90%
        """
        input_width = 28  # specifications for mnist dataset
        input_height = 28  # specifications for mnist dataset
        input_channels = 1  # specifications for mnist dataset

        # build a simple cnn
        kernel_sizes = [5]
        output_channels = [16]
        strides = [1]
        paddings = [2]
        hidden_dims_fully_connected = [64]
        output_dim = 10
        network = CNNValueNetwork(
            input_width=input_width,
            input_height=input_height,
            input_channels_count=input_channels,
            kernel_sizes=kernel_sizes,
            output_channels_list=output_channels,
            strides=strides,
            paddings=paddings,
            hidden_dims_fully_connected=hidden_dims_fully_connected,
            use_batch_norm_conv=True,  # testing to see if batch normalization works
            use_batch_norm_fully_connected=True,  # testing to see if batch normalization works
            output_dim=output_dim,
        )

        optimizer = optim.AdamW(
            network.parameters(), lr=self.learning_rate, amsgrad=True
        )
        criterion = torch.nn.CrossEntropyLoss()
        accuracy_train = 0.0

        # training loop
        accuracy_above_threshold = False
        for _ in range(self.num_epochs):
            # one epoch loops over entire dataset
            for x_batch, y_batch in self.train_dl:
                pred = network(x_batch)  # generate predictions
                loss = criterion(pred, y_batch)  # crossentropy loss for classification

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                is_correct = torch.argmax(pred, dim=1) == y_batch

                accuracy_train += is_correct.sum()

            accuracy_train /= len(self.train_dl.dataset)

            if accuracy_train > 0.9:
                accuracy_above_threshold = True
                break

        self.assertTrue(
            accuracy_above_threshold
        )  # training accuracy should reach above 0.9 with learning steps
