# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import List

import torch
import torch.nn.functional as F

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)


class OneHotActionTensorRepresentationModule(ActionRepresentationModule):
    """
    An one-hot action representation module.
    """

    def __init__(self, max_number_actions: int) -> None:
        super().__init__()
        self._max_number_actions = max_number_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        return (
            F.one_hot(x.long(), num_classes=self._max_number_actions)
            .squeeze(dim=-2)
            .float()
        )
        # (batch_size x action_dim)

    @property
    def max_number_actions(self) -> int:
        return self._max_number_actions

    @property
    def representation_dim(self) -> int:
        return self._max_number_actions

    def compare(self, other: ActionRepresentationModule) -> str:
        """
        Compares two OneHotActionTensorRepresentationModule instances for equality,
        checking the max_number_actions.

        Args:
          other: The other ActionRepresentationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, OneHotActionTensorRepresentationModule):
            differences.append(
                "other is not an instance of OneHotActionTensorRepresentationModule"
            )
        else:
            if self.max_number_actions != other.max_number_actions:
                differences.append(
                    f"max_number_actions is different: {self.max_number_actions} vs {other.max_number_actions}"
                )

        return "\n".join(differences)
