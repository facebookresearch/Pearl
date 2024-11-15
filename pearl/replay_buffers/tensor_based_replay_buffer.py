# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import random

from collections import deque
from typing import Deque, List, Optional, Tuple, Union

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch
from pearl.utils.device import get_default_device
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import Tensor


class TensorBasedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
    ) -> None:
        super(TensorBasedReplayBuffer, self).__init__()
        self.capacity = capacity
        # TODO: we want a unifying transition type
        self.memory: Deque[Union[Transition, TransitionBatch]] = deque(
            [], maxlen=capacity
        )
        self._device_for_batches: torch.device = get_default_device()

    def _store_transition(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        curr_available_actions_tensor_with_padding: Optional[Tensor],
        curr_unavailable_actions_mask: Optional[Tensor],
        next_state: Optional[SubjectiveState],
        next_available_actions_tensor_with_padding: Optional[Tensor],
        next_unavailable_actions_mask: Optional[Tensor],
        cost: Optional[float] = None,
    ) -> None:
        """
        Implements the way the replay buffer stores transitions.
        """
        raise NotImplementedError(f"{type(self)} has not implemented _store_transition")

    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool,
        curr_available_actions: Optional[ActionSpace] = None,
        next_state: Optional[SubjectiveState] = None,
        next_available_actions: Optional[ActionSpace] = None,
        max_number_actions: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> None:
        if self._is_action_continuous:
            (
                curr_available_actions_tensor_with_padding,
                curr_unavailable_actions_mask,
                next_available_actions_tensor_with_padding,
                next_unavailable_actions_mask,
            ) = (
                None,
                None,
                None,
                None,
            )
        else:
            (
                curr_available_actions_tensor_with_padding,
                curr_unavailable_actions_mask,
            ) = self.create_action_tensor_and_mask(
                max_number_actions, curr_available_actions
            )

            (
                next_available_actions_tensor_with_padding,
                next_unavailable_actions_mask,
            ) = self.create_action_tensor_and_mask(
                max_number_actions, next_available_actions
            )

            # Transitions require a "batch" dimension in the tensors,
            # so we add a dimension of size 1.
            if curr_available_actions_tensor_with_padding is not None:
                curr_available_actions_tensor_with_padding = (
                    curr_available_actions_tensor_with_padding.unsqueeze(0)
                )
            if curr_unavailable_actions_mask is not None:
                curr_unavailable_actions_mask = curr_unavailable_actions_mask.unsqueeze(
                    0
                )

            if next_available_actions_tensor_with_padding is not None:
                next_available_actions_tensor_with_padding = (
                    next_available_actions_tensor_with_padding.unsqueeze(0)
                )
            if next_unavailable_actions_mask is not None:
                next_unavailable_actions_mask = next_unavailable_actions_mask.unsqueeze(
                    0
                )

        self._store_transition(
            state,
            action,
            reward,
            terminated,
            curr_available_actions_tensor_with_padding,
            curr_unavailable_actions_mask,
            next_state,
            next_available_actions_tensor_with_padding,
            next_unavailable_actions_mask,
            cost,
        )

    @property
    def device_for_batches(self) -> torch.device:
        return self._device_for_batches

    @device_for_batches.setter
    def device_for_batches(self, new_device_for_batches: torch.device) -> None:
        self._device_for_batches = new_device_for_batches

    def _process_single_state(
        self, state: Optional[SubjectiveState]
    ) -> Optional[torch.Tensor]:
        if state is None:
            return None
        else:
            return self._process_non_optional_single_state(state)

    def _process_non_optional_single_state(
        self, state: SubjectiveState
    ) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            return state.to(get_default_device()).clone().detach().unsqueeze(0)
        else:
            return torch.tensor(state).unsqueeze(0)

    def _process_single_action(self, action: Action) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            return action.to(get_default_device()).clone().detach().unsqueeze(0)
        else:
            return torch.tensor(action).unsqueeze(0)

    def _process_single_reward(self, reward: Reward) -> torch.Tensor:
        return torch.tensor([reward])

    def _process_single_cost(self, cost: Optional[float]) -> Optional[torch.Tensor]:
        if cost is None:
            return None
        return torch.tensor([cost])

    def _process_single_terminated(self, terminated: bool) -> torch.Tensor:
        return torch.tensor([terminated])  # (1,)

    @staticmethod
    def create_action_tensor_and_mask(
        max_number_actions: Optional[int],
        available_action_space: Optional[ActionSpace],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Takes an action space containing only available actions,
        and the maximum number of actions possible.

        If the action space is continuous, returns (None, None).

        If the action space is discrete, returns a pair of tensors:

        1. A tensor of shape (action_space_size x action_dim) that contains the available
        actions.
        2. A mask tensor of shape (action_space_size) that contains 0 for available actions.

        Example:
        ----------------------------------------------------------
        Suppose the environment at every step has a maximum number of 5 actions, and
        the agent uses a onehot action representation module. At time step t, if the agent offers
        2 actions, [0, 3], then the result of this function will be:
        available_actions_tensor_with_padding = [
            [0],
            [3],
            [0],
            [0],
            [0],
        ]
        unavailable_actions_mask = [0, 0, 1, 1, 1]
        Note that although the actions and padding can overlap, the mask will always disable
        the unavailable actions so won't impact algorithm.

        The same goes to the case where the agent uses an identity action representation
        (assuming some random features for action 0 and 3), then it would be
        available_actions_tensor_with_padding = [
            [0.1, 0.6, 0.3, 1.8, 2.0],
            [0.8, -0.3, 0.6, 1.9, 3.0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        unavailable_actions_mask = [0, 0, 1, 1, 1]

        Args:
            max_number_actions (Optional[int]): The maximum number of actions possible.
            available_action_space (Optional[ActionSpace]): The action space containing
            only available actions.
        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: A pair of tensors
            containing the available actions and the mask of unavailable actions.
        """
        if max_number_actions is None or available_action_space is None:
            return (None, None)

        assert isinstance(available_action_space, DiscreteActionSpace)

        available_actions_tensor_with_padding = torch.zeros(
            (max_number_actions, available_action_space.action_dim),
            dtype=torch.float32,
        )  # (action_space_size x action_dim)
        available_actions_tensor = available_action_space.actions_batch
        available_actions_tensor_with_padding[: available_action_space.n, :] = (
            available_actions_tensor
        )

        unavailable_actions_mask = torch.zeros(
            (max_number_actions,)
        )  # (action_space_size)
        unavailable_actions_mask[available_action_space.n :] = 1
        unavailable_actions_mask = unavailable_actions_mask.bool()

        return (available_actions_tensor_with_padding, unavailable_actions_mask)

    def sample(self, batch_size: int) -> TransitionBatch:
        """
        The shapes of input and output are:
        input: batch_size

        output: TransitionBatch(
          state = tensor(batch_size, state_dim),
          action = tensor(batch_size, action_dim),
          reward = tensor(batch_size, ),
          next_state = tensor(batch_size, state_dim),
          curr_available_actions = tensor(batch_size, action_dim, action_dim),
          curr_available_actions_mask = tensor(batch_size, action_dim),
          next_available_actions = tensor(batch_size, action_dim, action_dim),
          next_available_actions_mask = tensor(batch_size, action_dim),
          terminated = tensor(batch_size, ),
        )
        """
        if batch_size > len(self):
            raise ValueError(
                f"Can't get a batch of size {batch_size} from a replay buffer with "
                f"only {len(self)} elements"
            )
        samples = random.sample(self.memory, batch_size)
        return self._create_transition_batch(
            # pyre-fixme[6]: For 1st argument expected `List[Transition]` but got
            #  `List[Union[Transition, TransitionBatch]]`.
            transitions=samples,
            is_action_continuous=self._is_action_continuous,
        )

    def __len__(self) -> int:
        return len(self.memory)

    def clear(self) -> None:
        self.memory = deque([], maxlen=self.capacity)

    def _create_transition_batch(
        self,
        transitions: List[Transition],
        is_action_continuous: bool,
    ) -> TransitionBatch:

        if len(transitions) == 0:
            return TransitionBatch(
                state=torch.empty(0),
                action=torch.empty(0),
                reward=torch.empty(0),
                next_state=torch.empty(0),
                next_action=torch.empty(0),
                curr_available_actions=torch.empty(0),
                curr_unavailable_actions_mask=torch.empty(0),
                next_available_actions=torch.empty(0),
                next_unavailable_actions_mask=torch.empty(0),
                terminated=torch.empty(0),
                cost=torch.empty(0),
            ).to(self.device_for_batches)

        has_next_state = transitions[0].next_state is not None
        has_next_action = transitions[0].next_action is not None
        has_curr_available_actions = transitions[0].curr_available_actions is not None
        has_next_available_actions = transitions[0].next_available_actions is not None
        has_cost_available = transitions[0].cost is not None

        state_list = []
        action_list = []
        reward_list = []
        cost_list = []
        terminated_list = []
        next_state_list = []
        next_action_list = []
        curr_available_actions_list = []
        curr_unavailable_actions_mask_list = []
        next_available_actions_list = []
        next_unavailable_actions_mask_list = []
        for x in transitions:
            state_list.append(x.state)
            action_list.append(x.action)
            reward_list.append(x.reward)
            terminated_list.append(x.terminated)
            if has_cost_available:
                cost_list.append(x.cost)
            if has_next_state:
                next_state_list.append(x.next_state)
            if has_next_action:
                next_action_list.append(x.next_action)
            if not is_action_continuous and has_curr_available_actions:
                curr_available_actions_list.append(x.curr_available_actions)
                curr_unavailable_actions_mask_list.append(
                    x.curr_unavailable_actions_mask
                )

            if not is_action_continuous and has_next_available_actions:
                next_available_actions_list.append(x.next_available_actions)
                next_unavailable_actions_mask_list.append(
                    x.next_unavailable_actions_mask
                )

        state_batch = torch.cat(state_list)
        action_batch = torch.cat(action_list)
        reward_batch = torch.cat(reward_list)
        terminated_batch = torch.cat(terminated_list)
        if has_cost_available:
            cost_batch = torch.cat(cost_list)
        else:
            cost_batch = None
        next_state_batch, next_action_batch = None, None
        if has_next_state:
            next_state_batch = torch.cat(next_state_list)
        if has_next_action:
            next_action_batch = torch.cat(next_action_list)
        curr_available_actions_batch, curr_unavailable_actions_mask_batch = None, None
        if not is_action_continuous and has_curr_available_actions:
            curr_available_actions_batch = torch.cat(curr_available_actions_list)
            curr_unavailable_actions_mask_batch = torch.cat(
                curr_unavailable_actions_mask_list
            )

        next_available_actions_batch, next_unavailable_actions_mask_batch = None, None
        if not is_action_continuous and has_next_available_actions:
            next_available_actions_batch = torch.cat(next_available_actions_list)
            next_unavailable_actions_mask_batch = torch.cat(
                next_unavailable_actions_mask_list
            )
        return TransitionBatch(
            state=state_batch,
            action=action_batch,
            reward=reward_batch,
            next_state=next_state_batch,
            next_action=next_action_batch,
            curr_available_actions=curr_available_actions_batch,
            curr_unavailable_actions_mask=curr_unavailable_actions_mask_batch,
            next_available_actions=next_available_actions_batch,
            next_unavailable_actions_mask=next_unavailable_actions_mask_batch,
            terminated=terminated_batch,
            cost=cost_batch,
        ).to(self.device_for_batches)
