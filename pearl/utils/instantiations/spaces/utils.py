# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from torch import Tensor


def reshape_to_1d_tensor(x: Tensor) -> Tensor:
    """
    Reshapes ``x`` to a 1-D tensor.

    Scalars are expanded and ``(1, d)`` tensors are squeezed. For tensors with
    more than one dimension, the elements are flattened.
    """
    if x.ndim == 1:
        return x
    if x.ndim == 0:  # scalar -> `d`
        return x.unsqueeze(dim=0)  # `1 x d` -> `d`
    if x.ndim == 2 and x.shape[0] == 1:
        return x.squeeze(dim=0)
    return x.flatten()
