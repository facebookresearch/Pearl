# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from enum import Enum


class TiebreakingStrategy(Enum):
    NO_TIEBREAKING = 0
    PER_ROW_TIEBREAKING = 1
    BATCH_TIEBREAKING = 2
