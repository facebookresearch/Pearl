# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from pearl.api.action import Action
from torch import Tensor


class Space(ABC):
    """An abstract base class for action and observation spaces in Pearl."""

    @abstractmethod
    def sample(self, mask: Optional[Tensor] = None) -> Action:
        """Samples an element from this space."""
        pass

    @property
    @abstractmethod
    def shape(self) -> torch.Size:
        """Returns the shape of an element of the space."""
        pass

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Checks whether this is a continuous space."""
        pass
