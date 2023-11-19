from __future__ import annotations

import logging

from typing import Iterator, List, Optional

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.utils.instantiations.action_spaces.utils import reshape_to_1d_tensor
from torch import Tensor

try:
    import gymnasium as gym
    from gymnasium.spaces import Discrete

    logging.info("Using 'gymnasium' package.")
except ModuleNotFoundError:
    import gym
    from gym.spaces import Discrete

    logging.warning("Using deprecated 'gym' package.")


class DiscreteActionSpace(ActionSpace):
    """A discrete action space containing finitely many `Action` objects.

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
        if len(actions) == 0:
            raise ValueError("`DiscreteActionSpace` requires at least one action.")
        self._set_validated_actions(actions=actions)  # sets self.actions
        self._gym_space = Discrete(n=len(actions), seed=seed, start=0)

    def _set_validated_actions(self, actions: List[Action]) -> None:
        """Creates the set of actions after validating that a action is a Tensor of
        shape `d` and all actions have the same shape."""
        # Allow scalar or (1, d) Tensors, but reshape them to (d,).
        # Use the first action's shape to compute the expected shape.
        validated_actions = []
        expected_shape = reshape_to_1d_tensor(actions[0]).shape
        for action in actions:
            action = reshape_to_1d_tensor(action)
            if action.shape != expected_shape:
                raise ValueError(
                    f"All actions must have the same shape. Expected {expected_shape}, "
                    f"but got {action.shape}."
                )
            validated_actions.append(action)
        self.actions = validated_actions

    @property
    def n(self) -> int:
        """Returns the number of actions in this action space."""
        return self._gym_space.n

    @property
    def is_continuous(self) -> bool:
        """Checks whether this is a continuous action space."""
        return False

    @property
    def action_dim(self) -> int:
        """Returns the dimensionality of an `Action` element from this space."""
        return self.actions[0].shape[0]

    @property
    def actions_batch(self) -> Tensor:
        """Returns a tensor of shape `(b, d)` with each row corresponding to an
        `Action` object from this action space."""
        return torch.stack(self.actions, dim=0)

    def sample(self, mask: Optional[Tensor] = None) -> Action:
        """Sample an action uniformly at random from this action space.

        Args:
            mask: An optional Tensor of shape `n` specifying the set of available
                actions, where `1` represents valid actions and `0` invalid actions.
                This mask is passed to Gymnasium's `Discrete.sample` method. If no
                actions are available, `self.actions[0]` is returned.

        Returns:
            A randomly sampled (available) action.
        """
        mask_np = mask.numpy().astype(int) if mask is not None else None
        action_idx = self._gym_space.sample(mask=mask_np)
        return self.actions[action_idx]

    def __iter__(self) -> Iterator[Action]:
        for action in self.actions:
            yield action

    def __getitem__(self, index: int) -> Action:
        return self.actions[index]

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
