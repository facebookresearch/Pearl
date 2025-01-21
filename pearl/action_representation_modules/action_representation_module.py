# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from abc import ABC, abstractmethod

import torch

import torch.nn as nn


class ActionRepresentationModule(ABC, nn.Module):
    """
    An abstract interface for action representation module.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compare(self, other: "ActionRepresentationModule") -> str:
        """
        Compares two action representation modules for equality.
        Note: subcomponents which are PyTorch modules are
        compared by state dict only.
        Args:
            other: The other action representation module to compare with.
        Returns:
            str: A string describing the differences, or an empty string if they are identical.
        """
        pass
