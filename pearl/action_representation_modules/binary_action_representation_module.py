# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

import torch

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)


class BinaryActionTensorRepresentationModule(ActionRepresentationModule):
    """
    Transform index to its binary representation.
    """

    def __init__(self, bits_num: int) -> None:
        super().__init__()
        self._bits_num = bits_num
        self._max_number_actions: int = 2**bits_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        return self.binary(x)
        # (batch_size x action_dim)

    def binary(self, x: torch.Tensor) -> torch.Tensor:
        mask = 2 ** torch.arange(self._bits_num).to(device=x.device)
        x = x.bitwise_and(mask).ne(0).byte()
        return x.to(dtype=torch.float32)

    @property
    def max_number_actions(self) -> int:
        return self._max_number_actions

    @property
    def representation_dim(self) -> int:
        return self._bits_num

    def compare(self, other: ActionRepresentationModule) -> str:
        """
        Compares two BinaryActionTensorRepresentationModule instances for equality,
        checking the bits_num and max_number_actions.

        Args:
          other: The other ActionRepresentationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, BinaryActionTensorRepresentationModule):
            differences.append(
                "other is not an instance of BinaryActionTensorRepresentationModule"
            )
        else:
            if self._bits_num != other._bits_num:
                differences.append(
                    f"bits_num is different: {self._bits_num} vs {other._bits_num}"
                )
            if self.max_number_actions != other.max_number_actions:
                differences.append(
                    f"max_number_actions is different: {self.max_number_actions} "
                    + f"vs {other.max_number_actions}"
                )

        return "\n".join(differences)
