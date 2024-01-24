# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import Tensor


def reshape_to_1d_tensor(x: Tensor) -> Tensor:
    """Reshapes a Tensor that is either scalar or `1 x d` -> `d`."""
    if x.ndim == 1:
        return x
    if x.ndim == 0:  # scalar -> `d`
        x = x.unsqueeze(dim=0)  # `1 x d` -> `d`
    elif x.ndim == 2 and x.shape[0] == 1:
        x = x.squeeze(dim=0)
    else:
        raise ValueError(f"Tensor of shape {x.shape} is not supported.")
    return x
