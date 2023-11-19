from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.utils.instantiations.action_spaces.utils import reshape_to_1d_tensor
from torch import Tensor

try:
    import gymnasium as gym
    from gymnasium.spaces import Box

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym
    from gym.spaces import Box

    logging.warning("Using deprecated 'gym' package.")


class BoxActionSpace(ActionSpace):
    """A continuous, box action space. This class is a wrapper around Gymnasium's
    `Box` space, but uses PyTorch tensors instead of NumPy arrays. The elements of
    this action space are assumed to be Tensors of shape `d`.
    """

    def __init__(
        self,
        low: Union[float, Tensor],
        high: Union[float, Tensor],
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> None:
        """Contructs a `BoxActionSpace`.

        Args:
            low: The lower bound on each dimension of the action space.
            high: The upper bound on each dimension of the action space.
            seed: Random seed used to initialize the random number generator of the
                underlying Gymnasium `Box` space.
        """
        low = (
            reshape_to_1d_tensor(low).numpy(force=True)
            if isinstance(low, Tensor)
            else np.array([low])
        )
        high = (
            reshape_to_1d_tensor(high).numpy(force=True)
            if isinstance(high, Tensor)
            else np.array([high])
        )
        self._gym_space = Box(low=low, high=high, seed=seed)

    @property
    def is_continuous(self) -> bool:
        """Checks whether this is a continuous action space."""
        return True

    @property
    def action_dim(self) -> int:
        """Returns the dimensionality of an `Action` element from this space."""
        return self._gym_space.shape[0]

    def sample(self, mask: Optional[Tensor] = None) -> Action:
        """Sample an action uniformly at random from this action space.

        Args:
            mask: An unused argument for the case of a `BoxActionSpace`, which
                does not support masking.

        Returns:
            A randomly sampled action.
        """
        if mask is not None:
            logging.warning(
                "Masked sampling is not supported in BoxActionSpace. Ignoring."
            )
        return torch.from_numpy(self._gym_space.sample())

    @property
    def low(self) -> Tensor:
        """Returns the lower bound of the action space."""
        return torch.from_numpy(self._gym_space.low)

    @property
    def high(self) -> Tensor:
        """Returns the upper bound of the action space."""
        return torch.from_numpy(self._gym_space.high)

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of an element of the space."""
        return torch.Size(self._gym_space.shape)

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
