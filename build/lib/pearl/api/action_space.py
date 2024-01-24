# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from abc import abstractmethod

from pearl.api.space import Space


class ActionSpace(Space):
    """An abstract base class for action spaces. An `ActionSpace` represents the set of
    all possible actions that can be taken by the agent. An `ActionSpace` should be
    specified such that each action is in its "environment representation," which means
    that an `action` from the space can be directly passed to `env.step(action)`.

    An `ActionSpace` should implement an `action_dim` method that returns the
    dimensionality of an `Action` from this space (which are 1-dim Tensors).
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Returns the dimensionality of an `Action` element from this space."""
        pass
