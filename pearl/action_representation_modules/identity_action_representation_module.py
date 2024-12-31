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


class IdentityActionRepresentationModule(ActionRepresentationModule):
    """
    An trivial class that outputs actions identitically as input.
    """

    def __init__(
        self,
        max_number_actions: int | None = None,
        representation_dim: int | None = None,
    ) -> None:
        super().__init__()
        self._max_number_actions = max_number_actions
        self._representation_dim = representation_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    # TODO: the properties below need to be at the ActionRepresentationModule level.
    # Several classes access them at that level, including PearlAgent.
    # Not sure why this is not generating Pyre errors.

    @property
    def max_number_actions(self) -> int | None:
        return self._max_number_actions

    @property
    def representation_dim(self) -> int | None:
        return self._representation_dim

    def compare(self, other: ActionRepresentationModule) -> str:
        """
        Compares two IdentityActionRepresentationModule instances for equality,
        checking max_number_actions and representation_dim.

        Args:
          other: The other ActionRepresentationModule to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        if not isinstance(other, IdentityActionRepresentationModule):
            differences.append(
                "other is not an instance of IdentityActionRepresentationModule"
            )
        else:
            if self.max_number_actions != other.max_number_actions:
                differences.append(
                    f"max_number_actions is different: {self.max_number_actions} "
                    + f"vs {other.max_number_actions}"
                )
            if self.representation_dim != other.representation_dim:
                differences.append(
                    f"representation_dim is different: {self.representation_dim} "
                    + f"vs {other.representation_dim}"
                )

        return "\n".join(differences)
