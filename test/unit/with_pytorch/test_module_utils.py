# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import torch
import torch.nn as nn
from later import unittest
from pearl.utils.module_utils import modules_have_similar_state_dict


class InnerModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


class OuterModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner1 = InnerModule()
        self.inner2 = InnerModule()


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm layer with buffers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Another BatchNorm layer
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc(x)
        return x


class TestModuleComparison(unittest.TestCase):
    def test_similar_modules(self) -> None:
        model1 = OuterModule()
        model2 = OuterModule()

        # Since weights are randomly initialized, make models identical
        model1.load_state_dict(model2.state_dict())

        self.assertEqual(modules_have_similar_state_dict(model1, model2), "")

    def test_different_modules(self) -> None:
        model1 = OuterModule()
        model2 = OuterModule()

        # Random initialization means that the models are different
        self.assertNotEqual(modules_have_similar_state_dict(model1, model2), "")

    def test_complex_model_with_same_buffers(self) -> None:
        model1 = CNN()
        model2 = CNN()

        # Since weights are randomly initialized, make models identical
        model1.load_state_dict(model2.state_dict())

        # Run a forward pass with THE SAME data to create identical buffers
        input_data = torch.randn(4, 3, 32, 32)
        model1(input_data)
        model2(input_data)  # Same input data

        self.assertEqual(modules_have_similar_state_dict(model1, model2), "")
