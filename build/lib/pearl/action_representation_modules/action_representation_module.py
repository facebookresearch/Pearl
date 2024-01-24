# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

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
