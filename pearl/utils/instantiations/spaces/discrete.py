# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from pearl.api.space import Space
from torch import Tensor

try:
    import gymnasium as gym
    from gymnasium.spaces import Discrete

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym
    from gym.spaces import Discrete

    logging.warning("Using deprecated 'gym' package.")


class DiscreteSpace(Space):
    """A discrete space containing finitely many elements.

    This class makes use of the `Discrete` space from Gymnasium, but uses an
    arbitrary list of Tensor objects instead of a range of integers.

    `DiscreteSpace` is also based on PyTorch tensors instead of NumPy arrays.
    """

    def __init__(self, elements: list[Tensor], seed: int | None = None) -> None:
        """Contructs a `DiscreteSpace`.

        Args:
            elements: A list of Tensors representing the elements of the space.
            seed: Random seed used to initialize the random number generator of the
                underlying Gym `Discrete` space.
        """
        if len(elements) == 0:
            raise ValueError("`DiscreteSpace` requires at least one element.")
        self._set_validated_elements(elements=elements)  # sets self.elements
        # pyre-fixme[28]: Unexpected keyword argument `start`.
        self._gym_space = Discrete(n=len(elements), seed=seed, start=0)

    def _set_validated_elements(self, elements: list[Tensor]) -> None:
        """Creates the set of elements after validating that they all have the
        same shape."""
        # Use the first shape to determine the expected shape.
        validated = []
        expected_shape = elements[0].shape
        for e in elements:
            if e.shape != expected_shape:
                raise ValueError(
                    f"All elements must have the same shape. Expected {expected_shape}, "
                    f"but got {e.shape}."
                )
            validated.append(e)
        self.elements = validated

    @property
    def n(self) -> int:
        """Returns the number of elements in this space."""
        return int(self._gym_space.n)

    @property
    def is_continuous(self) -> bool:
        """Checks whether this is a continuous space."""
        return False

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of an element of the space."""
        return self.elements[0].shape

    @property
    def gym_space(self) -> gym.Space:
        """Returns the underlying Gymnasium space."""
        # pyre-fixme[7]: Expected `Space[Any]` but got `Discrete`.
        return self._gym_space

    def sample(self, mask: Tensor | None = None) -> Tensor:
        """Sample an element uniformly at random from this space.

        Args:
            mask: An optional Tensor of shape `n` specifying the set of available
                elements, where `1` represents valid elements and `0` invalid elements.
                This mask is passed to Gymnasium's `Discrete.sample` method. If no
                elements are available, `self.elements[0]` is returned.

        Returns:
            A randomly sampled (available) element.
        """
        mask_np = mask.numpy().astype(int) if mask is not None else None
        # pyre-fixme[28]: Unexpected keyword argument `mask`.
        idx = self._gym_space.sample(mask=mask_np)
        return self.elements[idx]

    def __iter__(self) -> Iterator[Tensor]:
        yield from self.elements

    def __getitem__(self, index: int) -> Tensor:
        return self.elements[index]

    @staticmethod
    def from_gym(gym_space: gym.Space) -> DiscreteSpace:
        """Constructs a `DiscreteSpace` given a Gymnasium `Discrete` space.
        Convert from Gymnasium's index set {start, start + n - 1} to a list
        of tensors:
            [torch.tensor([start]), ..., torch.tensor([start + n - 1])],
        in accordance to what is expected by `DiscreteSpace`.

        Args:
            gym_space: A Gymnasium `Discrete` space.

        Returns:
            A `DiscreteSpace` with the same number of elements as `gym_space`.
        """
        assert isinstance(gym_space, Discrete)
        start, n = gym_space.start, gym_space.n
        return DiscreteSpace(
            elements=list(torch.arange(start=start, end=start + n).view(-1, 1)),
            seed=gym_space._np_random,
        )
