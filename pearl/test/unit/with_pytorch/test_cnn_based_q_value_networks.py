#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import numpy as np

import torch
import torchvision
from pearl.neural_networks.common.value_networks import CNNQValueNetwork

from torch.utils.data import DataLoader, Subset
from torchvision import transforms


class TestCNNQValueNetworks(unittest.TestCase):
    def setUp(self) -> None:
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_dataset = torchvision.datasets.MNIST(
            root="./", train=True, transform=transform, download=True
        )

        self.num_data_points = 5000
        self.mnist_train_dataset = Subset(
            mnist_dataset,
            np.arange(1, self.num_data_points),
        )
        self.learning_rate = 0.001
        self.batch_size = 64
        self.num_epochs = 1

        self.train_dl = DataLoader(
            self.mnist_train_dataset, self.batch_size, shuffle=True
        )

    def test_forward_pass(self) -> None:
        """
        test to check if the get_q_values method returns a scalar for a batch of
        images (observations) and a random batch of actions
        """
        input_width = 28  # specifications for mnist dataset
        input_height = 28  # specifications for mnist dataset
        input_channels = 1  # specifications for mnist dataset

        # build a cnn based q value network
        kernel_sizes = [5]
        output_channels = [16]
        strides = [1]
        paddings = [2]
        hidden_dims_fully_connected = [64]
        action_dim = 4  # random integer to generate dummy batch of actions
        network = CNNQValueNetwork(
            input_width=input_width,
            input_height=input_height,
            input_channels_count=input_channels,
            kernel_sizes=kernel_sizes,
            output_channels_list=output_channels,
            strides=strides,
            paddings=paddings,
            action_dim=action_dim,
            hidden_dims_fully_connected=hidden_dims_fully_connected,
        )

        for x_batch, _ in self.train_dl:
            action_batch = torch.rand((x_batch.shape[0], action_dim))
            q_values = network.get_q_values(
                x_batch, action_batch
            )  # test get_q_values method

            self.assertEqual(q_values.shape[0], x_batch.shape[0])
