# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch

import torch.testing as tt
from pearl.neural_networks.common.epistemic_neural_networks import Ensemble
from pearl.neural_networks.common.utils import ensemble_forward
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ...utils import create_normal_pdf_training_data


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
        self.network = Ensemble(
            input_dim=self.x_dim, hidden_dims=[64, 64], output_dim=1
        )

    def test_ensemble_values(self) -> None:
        """
        check that the values returned with for loop and vectorized implementation match
        """
        x = (
            self.train_dataset[0:15][0]
            .unsqueeze(1)
            .repeat(1, self.network.ensemble_size, 1)
        )
        for_loop_values = ensemble_forward(self.network.models, x, use_for_loop=True)
        vectorized_values = ensemble_forward(self.network.models, x, use_for_loop=False)
        self.assertEqual(for_loop_values.shape, vectorized_values.shape)
        tt.assert_close(
            for_loop_values,
            vectorized_values,
            atol=1e-5,
            rtol=0.0,
        )

    def test_ensemble_optimization(self) -> None:
        """
        ensemble should be able to fit a simple function and the loss value
        should get close to zero with a posterior variance > 1e-4
        NOTE: `use_for_loop=True` has to be used for optimization
        """

        optimizer = optim.AdamW(self.network.parameters(), self.learning_rate)
        criterion = torch.nn.MSELoss()
        losses = []

        for _ in range(self.num_epochs):
            # looping over entire dataset
            for x_batch, y_batch in self.train_dl:
                outputs = ensemble_forward(
                    self.network.models,
                    x_batch.unsqueeze(1).repeat(1, self.network.ensemble_size, 1),
                    use_for_loop=True,  # need to use for loop to backprop the gradients
                )
                loss_ensemble = criterion(
                    outputs, y_batch.unsqueeze(1).repeat(1, self.network.ensemble_size)
                )

                losses.append(loss_ensemble.item())

                optimizer.zero_grad()
                loss_ensemble.backward()
                optimizer.step()

        self.assertTrue(
            sum(losses[1:10]) > sum(losses[-10:])
        )  # loss should decrease over learning steps

        x = torch.normal(torch.zeros((1, self.x_dim)), torch.ones((1, self.x_dim)))
        x = x.unsqueeze(1).repeat(1, self.network.ensemble_size, 1)
        variance = torch.var(ensemble_forward(self.network.models, x))

        self.assertTrue(variance > 1e-6)
