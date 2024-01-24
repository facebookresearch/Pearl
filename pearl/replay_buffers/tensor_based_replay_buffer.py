# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

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


class TensorBasedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        has_next_state: bool = True,
        has_next_action: bool = True,
        has_next_available_actions: bool = True,
        has_cost_available: bool = False,
    ) -> None:
        super(TensorBasedReplayBuffer, self).__init__()
        self.capacity = capacity
        # TODO: we want a unifying transition type
        self.memory: Deque[Union[Transition, TransitionBatch]] = deque(
            [], maxlen=capacity
        )
        self._has_next_state = has_next_state
        self._has_next_action = has_next_action
        self._has_next_available_actions = has_next_available_actions
        self.has_cost_available = has_cost_available
        self._device: torch.device = get_default_device()

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: torch.device) -> None:
        self._device = value

    def _process_single_state(self, state: SubjectiveState) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            return state.clone().detach().to(self._device).unsqueeze(0)
        else:
            return torch.tensor(state, device=self._device).unsqueeze(0)

    def _process_single_action(self, action: Action) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            return action.clone().detach().to(self._device).unsqueeze(0)
        else:
            return torch.tensor(action, device=self._device).unsqueeze(0)

    def _process_single_reward(self, reward: Reward) -> torch.Tensor:
        return torch.tensor([reward], device=self._device)

    def _process_single_cost(self, cost: Optional[float]) -> Optional[torch.Tensor]:
        if cost is None:
            return None
        return torch.tensor([cost], device=self._device)

    def _process_single_done(self, done: bool) -> torch.Tensor:
        return torch.tensor([done], device=self._device)  # (1,)

    """
    This function is only used for discrete action space.
    An example:
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
    Note that although the actions and padding can have overlap, the mask will always disable the
    unavailable actions so won't impact algorithm.

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
    """

    def _create_action_tensor_and_mask(
        self, max_number_actions: Optional[int], available_action_space: ActionSpace
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._is_action_continuous or max_number_actions is None:
            return (None, None)

        assert isinstance(available_action_space, DiscreteActionSpace)

        available_actions_tensor_with_padding = torch.zeros(
            (1, max_number_actions, available_action_space.action_dim),
            device=self._device,
            dtype=torch.float32,
        )  # (1 x action_space_size x action_dim)
        available_actions_tensor = available_action_space.actions_batch
        available_actions_tensor_with_padding[
            0, : available_action_space.n, :
        ] = available_actions_tensor

        unavailable_actions_mask = torch.zeros(
            (1, max_number_actions), device=self._device
        )  # (1 x action_space_size)
        unavailable_actions_mask[0, available_action_space.n :] = 1
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
          done = tensor(batch_size, ),
        )
        """
        if batch_size > len(self):
            raise ValueError(
                f"Can't get a batch of size {batch_size} from a replay buffer with"
                f"only {len(self)} elements"
            )
        samples = random.sample(self.memory, batch_size)
        return self._create_transition_batch(
            transitions=samples,
            has_next_state=self._has_next_state,
            has_next_action=self._has_next_action,
            is_action_continuous=self._is_action_continuous,
            has_next_available_actions=self._has_next_available_actions,
            has_cost_available=self.has_cost_available,
        )

    def __len__(self) -> int:
        return len(self.memory)

    def clear(self) -> None:
        self.memory = deque([], maxlen=self.capacity)

    def _create_transition_batch(
        self,
        transitions: List[Transition],
        has_next_state: bool,
        has_next_action: bool,
        is_action_continuous: bool,
        has_next_available_actions: bool,
        has_cost_available: bool,
    ) -> TransitionBatch:
        state_list = []
        action_list = []
        reward_list = []
        cost_list = []
        done_list = []
        cum_reward_list = []
        cum_reward_batch = 0
        next_state_list = []
        next_action_list = []
        curr_available_actions_list = []
        curr_unavailable_actions_mask_list = []
        next_available_actions_list = []
        next_unavailable_actions_mask_list = []
        has_none_cum_reward = False
        for x in transitions:
            state_list.append(x.state)
            action_list.append(x.action)
            reward_list.append(x.reward)
            done_list.append(x.done)
            if has_cost_available:
                cost_list.append(x.cost)
            if x.cum_reward is not None:
                cum_reward_list.append(x.cum_reward)
            else:
                has_none_cum_reward = True
            if has_next_state:
                next_state_list.append(x.next_state)
            if has_next_action:
                next_action_list.append(x.next_action)
            if not is_action_continuous:
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
        done_batch = torch.cat(done_list)
        cum_reward_batch = None
        if has_cost_available:
            cost_batch = torch.cat(cost_list)
        else:
            cost_batch = None
        if not has_none_cum_reward:
            cum_reward_batch = torch.cat(cum_reward_list)
        next_state_batch, next_action_batch = None, None
        if has_next_state:
            next_state_batch = torch.cat(next_state_list)
        if has_next_action:
            next_action_batch = torch.cat(next_action_list)
        curr_available_actions_batch, curr_unavailable_actions_mask_batch = None, None
        if not is_action_continuous:
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
            done=done_batch,
            cum_reward=cum_reward_batch,
            cost=cost_batch,
        ).to(self.device)
