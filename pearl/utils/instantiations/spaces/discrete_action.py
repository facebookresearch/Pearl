# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import logging

from typing import List, Optional

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace
from pearl.utils.instantiations.spaces.utils import reshape_to_1d_tensor
from torch import Tensor

try:
    import gymnasium as gym
    from gymnasium.spaces import Discrete

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym
    from gym.spaces import Discrete

    logging.warning("Using deprecated 'gym' package.")


class DiscreteActionSpace(DiscreteSpace, ActionSpace):
    """A discrete space containing finitely many `Action` objects. This is a
    special case of a `DiscreteSpace` that performs some shape checking to make
    sure that each action is a Tensor of shape `(d,)`.

    This class makes use of the `Discrete` space from Gymnasium, but uses an
    arbitrary list of `Action` objects instead of a range of integers.

    `DiscreteActionSpace` is based on PyTorch tensors instead of NumPy arrays.
    """

    def __init__(self, actions: List[Action], seed: Optional[int] = None) -> None:
        """Contructs a `DiscreteActionSpace`.

        Args:
            actions: A list of possible `Action` objects.
            seed: Random seed used to initialize the random number generator of the
                underlying Gym `Discrete` space.
        """
        super(DiscreteActionSpace, self).__init__(elements=actions, seed=seed)

    def _set_validated_elements(self, elements: List[Tensor]) -> None:
        """Creates the set of actions after validating that a action is a Tensor of
        shape `d` and all actions have the same shape."""
        # Allow scalar or (1, d) Tensors, but reshape them to (d,).
        # Use the first action's shape to compute the expected shape.
        validated_actions = []
        expected_shape = reshape_to_1d_tensor(elements[0]).shape
        for action in elements:
            action = reshape_to_1d_tensor(action)
            if action.shape != expected_shape:
                raise ValueError(
                    f"All actions must have the same shape. Expected {expected_shape}, "
                    f"but got {action.shape}."
                )
            validated_actions.append(action)
        self.elements = validated_actions

    @property
    def actions(self) -> List[Action]:
        """Returns the list of possible `Action` objects."""
        return self.elements

    @property
    def actions_batch(self) -> Tensor:
        """Returns a tensor of shape `(b, d)` with each row corresponding to an
        `Action` object from this action space."""
        return torch.stack(self.actions, dim=0)

    @property
    def action_dim(self) -> int:
        """Returns the dimensionality of an `Action` element from this space."""
        return self.shape[0]

    @staticmethod
    def from_gym(gym_space: gym.Space) -> DiscreteActionSpace:
        """Constructs a `DiscreteActionSpace` given a Gymnasium `Discrete` space.
        Convert from Gymnasium's action index set {start, start + n - 1} to a list
        of action tensors:
            [torch.tensor([start]), ..., torch.tensor([start + n - 1])],
        in accordance to what is expected by `DiscreteActionSpace`.

        Args:
            gym_space: A Gymnasium `Discrete` space.

        Returns:
            A `DiscreteActionSpace` with the same number of actions as `gym_space`.
        """
        assert isinstance(gym_space, Discrete)
        start, n = gym_space.start, gym_space.n
        return DiscreteActionSpace(
            actions=list(torch.arange(start=start, end=start + n).view(-1, 1)),
            seed=gym_space._np_random,
        )

    def to(self, device: torch.device) -> None:
        for i, action in enumerate(self.actions):
            self.actions[i] = action.to(device)
