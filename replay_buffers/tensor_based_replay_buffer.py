import random
from collections import deque

import torch
import torch.nn.functional as F

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.device import get_pearl_device


class TensorBasedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        has_next_state: bool = True,
        has_next_action: bool = True,
        has_next_available_actions=True,
    ) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self._has_next_state = has_next_state
        self._has_next_action = has_next_action
        self._has_next_available_actions = has_next_available_actions
        # TODO we could add to init input if needed in the future
        self.is_action_continuous = False
        self.device = get_pearl_device()

    def _process_single_state(self, state: SubjectiveState) -> torch.tensor:
        return torch.tensor(state, device=self.device).unsqueeze(0)  # (1 x state_dim)

    def _process_single_action(
        self, action: Action, action_space: ActionSpace
    ) -> torch.tensor:
        if self.is_action_continuous:
            return torch.tensor(action, device=self.device).reshape(
                1, -1
            )  # (1 x action_dim)
        return F.one_hot(
            torch.tensor([action], device=self.device), num_classes=action_space.n
        )  # (1 x action_dim)

    def _process_single_reward(self, reward: float) -> torch.tensor:
        return torch.tensor([reward], device=self.device).float()

    def _process_single_done(self, done: bool) -> torch.tensor:
        return torch.tensor([done], device=self.device).float()  # (1)

    def _create_action_tensor_and_mask(
        self, action_space: ActionSpace, available_actions: ActionSpace
    ) -> (torch.tensor, torch.tensor):
        if self.is_action_continuous:
            return (None, None)  # continuous action does not have limited space
        available_actions_tensor_with_padding = torch.zeros(
            (1, action_space.n, action_space.n), device=self.device
        )  # (1 x action_space_size x action_dim)
        available_actions_tensor = F.one_hot(
            torch.arange(0, available_actions.n, device=self.device),
            num_classes=action_space.n,
        )  # (1 x available_action_space_size x action_dim)
        available_actions_tensor_with_padding[
            0, : available_actions.n, :
        ] = available_actions_tensor
        available_actions_mask = torch.zeros(
            (1, action_space.n), device=self.device
        )  # (1 x action_space_size)
        available_actions_mask[0, available_actions.n :] = 1
        available_actions_mask = available_actions_mask.bool()

        return (available_actions_tensor_with_padding, available_actions_mask)

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
                f"Can't get a batch of size {batch_size} from a replay buffer with only {len(self)} elements"
            )
        samples = random.sample(self.memory, batch_size)
        return TransitionBatch(
            state=torch.cat([x.state for x in samples]),
            action=torch.cat([x.action for x in samples]),
            reward=torch.cat([x.reward for x in samples]),
            next_state=torch.cat([x.next_state for x in samples])
            if self._has_next_state
            else None,
            next_action=torch.cat([x.next_action for x in samples])
            if self._has_next_action
            else None,
            curr_available_actions=torch.cat(
                [x.curr_available_actions for x in samples]
            )
            if not self.is_action_continuous
            else None,
            curr_available_actions_mask=torch.cat(
                [x.curr_available_actions_mask for x in samples]
            )
            if not self.is_action_continuous
            else None,
            next_available_actions=torch.cat(
                [x.next_available_actions for x in samples]
            )
            if not self.is_action_continuous and self._has_next_available_actions
            else None,
            next_available_actions_mask=torch.cat(
                [x.next_available_actions_mask for x in samples]
            )
            if not self.is_action_continuous and self._has_next_available_actions
            else None,
            done=torch.cat([x.done for x in samples]),
        )

    def __len__(self) -> int:
        return len(self.memory)

    def empty(self) -> None:
        self.memory = deque([], maxlen=self.capacity)
