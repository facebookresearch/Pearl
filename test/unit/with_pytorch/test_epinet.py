# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import copy
import unittest

import torch

import torch.testing as tt
from pearl.neural_networks.common.epistemic_neural_networks import Epinet
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ...utils import create_normal_pdf_training_data


class TestEpinet(unittest.TestCase):
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
        self.train_dl = DataLoader(
            self.train_dataset, self.batch_size, shuffle=True, num_workers=4
        )
        self.network = Epinet(
            index_dim=10,
            input_dim=self.x_dim,
            output_dim=1,
            num_indices=10,
            epi_hiddens=[64, 64],
            prior_hiddens=[16, 16],
            prior_scale=0.3,
        )

    def test_priornet_values(self) -> None:
        """
        check that the values returned with for loop and vectorized implementation match
        """
        x = self.train_dataset[0:15][0]
        z = torch.normal(
            torch.zeros(self.network.index_dim),
            torch.ones(self.network.num_indices, self.network.index_dim),
        )
        xz = self.network.format_xz(x, z)
        x_cartesian, z_cartesian = (
            xz[:, : -self.network.index_dim],
            xz[:, -self.network.index_dim :],
        )
        vectorized_values = self.network.priornet(x_cartesian, z_cartesian)

        for_loop_values = []
        for i in range(self.network.index_dim):
            model = self.network.priornet.models[i]
            output = model(x_cartesian)
            for_loop_values.append(output * z_cartesian[:, i : i + 1])
        for_loop_values = torch.sum(
            torch.cat(for_loop_values, dim=1), dim=1, keepdim=True
        )

        self.assertEqual(for_loop_values.shape, vectorized_values.shape)
        tt.assert_close(
            for_loop_values,
            vectorized_values,
            atol=1e-5,
            rtol=0.0,
        )

    def test_epinet_optimization(self) -> None:
        """
        epinet should be able to fit a simple function and the loss value
        should get close to zero with a posterior variance > 1e-6
        """

        optimizer = optim.AdamW(self.network.parameters(), self.learning_rate)
        criterion = torch.nn.MSELoss()
        losses = []

        for _ in range(self.num_epochs):
            # looping over entire dataset
            for x_batch, y_batch in self.train_dl:
                z_batch = torch.normal(
                    torch.zeros(self.network.index_dim),
                    torch.ones(self.network.num_indices, self.network.index_dim),
                )
                yz = self.network.format_xz(y_batch.unsqueeze(dim=1), z_batch)
                y_cartesian = yz[:, : -self.network.index_dim]
                outputs = self.network(x_batch, z_batch)
                loss_epinet = criterion(outputs, y_cartesian)

                losses.append(loss_epinet.item())

                optimizer.zero_grad()
                loss_epinet.backward()
                optimizer.step()

        self.assertTrue(
            sum(losses[1:10]) > sum(losses[-10:])
        )  # loss should decrease over learning steps

        x = torch.normal(torch.zeros((1, self.x_dim)), torch.ones((1, self.x_dim)))
        z = torch.normal(
            torch.zeros(self.network.index_dim),
            torch.ones(1000, self.network.index_dim),
        )
        variance = torch.var(self.network(x, z))

        self.assertTrue(variance > 1e-6)

    def test_priornet_constant(self) -> None:
        """
        The priornet should not be updated during training.
        Ensure that the epinet weights are updated and priornet weights are not.
        """

        optimizer = optim.AdamW(self.network.parameters(), self.learning_rate)
        criterion = torch.nn.MSELoss()
        losses = []

        init_priornet_weight = copy.deepcopy(
            self.network.priornet.models[0][0][0].weight
        )
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        init_epinet_weight = copy.deepcopy(self.network.epinet[0][0].weight)

        for _ in range(self.num_epochs):
            # looping over entire dataset
            for x_batch, y_batch in self.train_dl:
                z_batch = torch.normal(
                    torch.zeros(self.network.index_dim),
                    torch.ones(self.network.num_indices, self.network.index_dim),
                )
                yz = self.network.format_xz(y_batch.unsqueeze(dim=1), z_batch)
                y_cartesian = yz[:, : -self.network.index_dim]
                outputs = self.network(x_batch, z_batch)
                loss_epinet = criterion(outputs, y_cartesian)

                losses.append(loss_epinet.item())

                optimizer.zero_grad()
                loss_epinet.backward()
                optimizer.step()

        self.assertTrue(
            sum(losses[1:10]) > sum(losses[-10:])
        )  # loss should decrease over learning steps
        final_priornet_weight = copy.deepcopy(
            self.network.priornet.models[0][0][0].weight
        )
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        final_epinet_weight = copy.deepcopy(self.network.epinet[0][0].weight)
        tt.assert_close(
            final_priornet_weight,
            init_priornet_weight,
            atol=1e-5,
            rtol=0.0,
        )
        self.assertTrue(
            torch.linalg.matrix_norm(final_epinet_weight - init_epinet_weight) > 1e-2
        )

    def test_epinet_save_load(self) -> None:
        """
        Ensure that the model can be saved and loaded correctly.
        """
        scripted_module = torch.jit.script(
            Epinet(
                index_dim=10,
                input_dim=4,
                output_dim=1,
                num_indices=10,
                epi_hiddens=[64, 64],
                prior_hiddens=[16, 16],
                prior_scale=0.3,
            )
        )

        x = torch.normal(torch.zeros((1, self.x_dim)), torch.ones((1, self.x_dim)))
        z = torch.normal(
            torch.zeros(scripted_module.index_dim),
            torch.ones(1, scripted_module.index_dim),
        )
        output = scripted_module(x, z).detach()
        torch.jit.save(scripted_module, "epinet.pt")
        loaded_model = torch.jit.load("epinet.pt")
        loaded_output = loaded_model(x, z).detach()
        self.assertTrue(torch.allclose(output, loaded_output))
