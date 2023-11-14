import random
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch
from pearl.utils.device import get_default_device


class TensorBasedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        has_next_state: bool = True,
        has_next_action: bool = True,
        # pyre-fixme[2]: Parameter must be annotated.
        has_next_available_actions=True,
    ) -> None:
        super(TensorBasedReplayBuffer, self).__init__()
        self.capacity = capacity
        # pyre-fixme[4]: Attribute must be annotated.
        self.memory = deque([], maxlen=capacity)
        self._has_next_state = has_next_state
        self._has_next_action = has_next_action
        # pyre-fixme[4]: Attribute must be annotated.
        self._has_next_available_actions = has_next_available_actions
        self._device: torch.device = get_default_device()

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: torch.device) -> None:
        self._device = value

    def _process_single_state(self, state: SubjectiveState) -> torch.Tensor:
        return torch.tensor(state, device=self._device).unsqueeze(0)

    def _process_single_action(
        self, action: Action, action_space: ActionSpace
    ) -> torch.Tensor:
        if self._is_action_continuous:
            return torch.tensor(action, device=self._device).reshape(
                1, -1
            )  # (1 x action_dim)
        return F.one_hot(
            torch.tensor([action], device=self._device),
            # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
            num_classes=action_space.n,
        )  # (1 x action_dim)

    def _process_single_reward(self, reward: float) -> torch.Tensor:
        return torch.tensor([reward], device=self._device)

    def _process_single_done(self, done: bool) -> torch.Tensor:
        return torch.tensor([done], device=self._device)  # (1,)

    def _create_action_tensor_and_mask(
        self, action_space: ActionSpace, available_actions: ActionSpace
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._is_action_continuous:
            return (None, None)  # continuous action does not have limited space
        available_actions_tensor_with_padding = torch.zeros(
            # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
            (1, action_space.n, action_space.n),
            device=self._device,
        )  # (1 x action_space_size x action_dim)
        available_actions_tensor = F.one_hot(
            torch.arange(0, available_actions.n, device=self._device),
            num_classes=action_space.n,
        )  # (1 x available_action_space_size x action_dim)
        available_actions_tensor_with_padding[
            0, : available_actions.n, :
        ] = available_actions_tensor
        available_actions_mask = torch.zeros(
            (1, action_space.n), device=self._device
        )  # (1 x action_space_size)
        available_actions_mask[0, available_actions.n :] = 1
        available_actions_mask = available_actions_mask.bool()

        return (available_actions_tensor_with_padding, available_actions_mask)

    # pyre-fixme[15]: `sample` overrides method defined in `ReplayBuffer`
    #  inconsistently.
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
    ) -> TransitionBatch:
        # TODO[drjiang]: Will properly handle the None pyre errors in this function,
        # in a subsequent diff. Errors are due to parts of the transition potentially
        # being None.
        state_list = []
        action_list = []
        reward_list = []
        done_list = []
        cum_reward_list = []
        cum_reward_batch = 0
        next_state_list = []
        next_action_list = []
        curr_available_actions_list = []
        curr_available_actions_mask_list = []
        next_available_actions_list = []
        next_available_actions_mask_list = []
        has_none_cum_reward = False
        for x in transitions:
            state_list.append(x.state)
            action_list.append(x.action)
            reward_list.append(x.reward)
            done_list.append(x.done)
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
                curr_available_actions_mask_list.append(x.curr_available_actions_mask)

            if not is_action_continuous and has_next_available_actions:
                next_available_actions_list.append(x.next_available_actions)
                next_available_actions_mask_list.append(x.next_available_actions_mask)

        state_batch = torch.cat(state_list)
        action_batch = torch.cat(action_list)
        reward_batch = torch.cat(reward_list)
        done_batch = torch.cat(done_list)
        cum_reward_batch = None
        if not has_none_cum_reward:
            cum_reward_batch = torch.cat(cum_reward_list)
        next_state_batch, next_action_batch = None, None
        if has_next_state:
            next_state_batch = torch.cat(next_state_list)
        if has_next_action:
            next_action_batch = torch.cat(next_action_list)
        curr_available_actions_batch, curr_available_actions_mask_batch = None, None
        if not is_action_continuous:
            curr_available_actions_batch = torch.cat(curr_available_actions_list)
            curr_available_actions_mask_batch = torch.cat(
                curr_available_actions_mask_list
            )

        next_available_actions_batch, next_available_actions_mask_batch = None, None
        if not is_action_continuous and has_next_available_actions:
            next_available_actions_batch = torch.cat(next_available_actions_list)
            next_available_actions_mask_batch = torch.cat(
                next_available_actions_mask_list
            )
        return TransitionBatch(
            state=state_batch,
            action=action_batch,
            reward=reward_batch,
            next_state=next_state_batch,
            next_action=next_action_batch,
            curr_available_actions=curr_available_actions_batch,
            curr_available_actions_mask=curr_available_actions_mask_batch,
            next_available_actions=next_available_actions_batch,
            next_available_actions_mask=next_available_actions_mask_batch,
            done=done_batch,
            cum_reward=cum_reward_batch,
        ).to(self.device)
