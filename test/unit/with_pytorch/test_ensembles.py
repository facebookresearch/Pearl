#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.core.common.neural_networks.epistemic_neural_networks import Ensemble
from pearl.core.common.neural_networks.utils import ensemble_forward

from pearl.test.utils import create_normal_pdf_training_data
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


class TestEnsembles(unittest.TestCase):
    def setUp(self) -> None:
        self.x_dim = 4
        self.num_data_points = 1000
        self.learning_rate = 0.001
        self.batch_size = 100
        self.num_epochs = 10

        # create a data set of (x, y) and form a dataloader
        x, y = create_normal_pdf_training_data(
            input_dim=self.x_dim, num_data_points=self.num_data_points
        )
        self.train_dataset = TensorDataset(x, y)
        self.train_dl = DataLoader(self.train_dataset, self.batch_size, shuffle=True)

    def test_ensemble(self) -> None:
        """
        ensemble should be able to fit a simple function and the loss value
        should get close to zero with a posterior variance > 1e-4
        """

        network = Ensemble(input_dim=self.x_dim, hidden_dims=[64, 64], output_dim=1)
        optimizer = optim.AdamW(network.parameters(), self.learning_rate)
        criterion = torch.nn.MSELoss()
        losses = []

        for _ in range(self.num_epochs):
            # looping over entire dataset
            for x_batch, y_batch in self.train_dl:
                loss_ensemble = torch.tensor(0.0)
                # ensemble_forward does not pass gradients
                # need to loop through for gradient steps
                for z in range(network.ensemble_size):
                    outputs = network(x_batch, torch.tensor(z))
                    loss = criterion(outputs, y_batch)
                    loss_ensemble += loss

                losses.append(loss_ensemble.item())

                optimizer.zero_grad()
                loss_ensemble.backward()
                optimizer.step()

        self.assertTrue(
            sum(losses[1:10]) > sum(losses[-10:])
        )  # loss should decrease over learning steps

        x = torch.normal(torch.zeros((1, self.x_dim)), torch.ones((1, self.x_dim)))
        x = x.unsqueeze(1).repeat(1, network.ensemble_size, 1)
        variance = torch.var(ensemble_forward(network.models, x))

        self.assertTrue(variance > 1e-6)
