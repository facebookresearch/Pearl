# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

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
        super(OneHotActionTensorRepresentationModule, self).__init__()
        self._max_number_actions = max_number_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(x.long(), num_classes=self._max_number_actions).squeeze(dim=-2)
        # (batch_size x action_dim)

    @property
    def max_number_actions(self) -> int:
        return self._max_number_actions

    @property
    def representation_dim(self) -> int:
        return self._max_number_actions
