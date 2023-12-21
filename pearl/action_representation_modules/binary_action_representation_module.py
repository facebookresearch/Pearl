# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)


class BinaryActionTensorRepresentationModule(ActionRepresentationModule):
    """
    Transform index to its binary representation.
    """

    def __init__(self, bits_num: int) -> None:
        super(BinaryActionTensorRepresentationModule, self).__init__()
        self._bits_num = bits_num
        self._max_number_actions: int = 2**bits_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
