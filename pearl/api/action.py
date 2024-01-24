# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import Tensor

# An `Action` is expected to be a 1-dim Tensor of shape `(d,)`, where `d` is the
# is the dimensionality of the action.
Action = Tensor
