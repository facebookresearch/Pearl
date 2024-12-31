# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from __future__ import annotations

import logging

import numpy as np
import torch
from pearl.api.action_space import ActionSpace
from pearl.utils.instantiations.spaces.box import BoxSpace
from pearl.utils.instantiations.spaces.utils import reshape_to_1d_tensor
from torch import Tensor

try:
    import gymnasium as gym
    from gymnasium.spaces import Box

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym
    from gym.spaces import Box

    logging.warning("Using deprecated 'gym' package.")


class BoxActionSpace(BoxSpace, ActionSpace):
    """A continuous, box action space. This class takes most of the functionality
    from `BoxSpace`, but there are two differences. First, the elements of this
    action space are assumed to be Tensors of shape `d`. Second, an `action_dim`
    convenience method is implemented.
    """

    def __init__(
        self,
        low: float | Tensor,
        high: float | Tensor,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """Contructs a `BoxActionSpace`.

        Args:
            low: The lower bound on each dimension of the action space.
            high: The upper bound on each dimension of the action space.
            seed: Random seed used to initialize the random number generator of the
                underlying Gymnasium `Box` space.
        """
        # pyre-fixme[9]: low has type `Union[float, Tensor]`; used as `ndarray[Any,
        #  Any]`.
        low = (
            reshape_to_1d_tensor(low).numpy(force=True)
            if isinstance(low, Tensor)
            else np.array([low])
        )
        # pyre-fixme[9]: high has type `Union[float, Tensor]`; used as `ndarray[Any,
        #  Any]`.
        high = (
            reshape_to_1d_tensor(high).numpy(force=True)
            if isinstance(high, Tensor)
            else np.array([high])
        )
        super().__init__(low=low, high=high, seed=seed)

    @property
    def action_dim(self) -> int:
        """Returns the dimensionality of an `Action` element from this space."""
        return self.shape[-1]

    @staticmethod
    def from_gym(gym_space: gym.Space) -> BoxActionSpace:
        """Constructs a `BoxActionSpace` given a Gymnasium `Box` space.

        Args:
            gym_space: A Gymnasium `Box` space.

        Returns:
            A `BoxActionSpace` with the same bounds and seed as `gym_space`.
        """
        assert isinstance(gym_space, Box)
        return BoxActionSpace(
            low=torch.from_numpy(gym_space.low),
            high=torch.from_numpy(gym_space.high),
            seed=gym_space._np_random,
        )
