import random
from collections import deque
from typing import List

import torch
import torch.nn.functional as F

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch
from pearl.utils.device import get_pearl_device


class TensorBasedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        has_next_state: bool = True,
        has_next_action: bool = True,
        # pyre-fixme[2]: Parameter must be annotated.
        has_next_available_actions=True,
    ) -> None:
        self.capacity = capacity
        # pyre-fixme[4]: Attribute must be annotated.
        self.memory = deque([], maxlen=capacity)
        self._has_next_state = has_next_state
        self._has_next_action = has_next_action
        # pyre-fixme[4]: Attribute must be annotated.
        self._has_next_available_actions = has_next_available_actions
        # TODO we could add to init input if needed in the future
        self.is_action_continuous = False
        # pyre-fixme[4]: Attribute must be annotated.
        self.device = get_pearl_device()

    # pyre-fixme[11]: Annotation `tensor` is not defined as a type.
    def _process_single_state(self, state: SubjectiveState) -> torch.tensor:
        return (
            torch.tensor(state, device=self.device).unsqueeze(0).float()
        )  # (1 x state_dim)

    def _process_single_action(
        self, action: Action, action_space: ActionSpace
    ) -> torch.tensor:
        if self.is_action_continuous:
            return torch.tensor(action, device=self.device).reshape(
                1, -1
            )  # (1 x action_dim)
        return F.one_hot(
            torch.tensor([action], device=self.device),
            # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
            num_classes=action_space.n,
        )  # (1 x action_dim)

    def _process_single_reward(self, reward: float) -> torch.tensor:
        return torch.tensor([reward], device=self.device).float()

    def _process_single_done(self, done: bool) -> torch.tensor:
        return torch.tensor([done], device=self.device).float()  # (1)

    def _create_action_tensor_and_mask(
        self,
        action_space: ActionSpace,
        available_actions: ActionSpace
        # pyre-fixme[31]: Expression `tensor)` is not a valid type.
    ) -> (torch.tensor, torch.tensor):
        if self.is_action_continuous:
            return (None, None)  # continuous action does not have limited space
        available_actions_tensor_with_padding = torch.zeros(
            # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
            (1, action_space.n, action_space.n),
            device=self.device,
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
                f"Can't get a batch of size {batch_size} from a replay buffer with only {len(self)} elements"
            )
        samples = random.sample(self.memory, batch_size)
        return _create_transition_batch(
            transitions=samples,
            has_next_state=self._has_next_state,
            has_next_action=self._has_next_action,
            is_action_continuous=self.is_action_continuous,
            has_next_available_actions=self._has_next_available_actions,
        )

    def __len__(self) -> int:
        return len(self.memory)

    def empty(self) -> None:
        self.memory = deque([], maxlen=self.capacity)


def _create_transition_batch(
    transitions: List[Transition],
    has_next_state: bool,
    has_next_action: bool,
    is_action_continuous: bool,
    has_next_available_actions: bool,
) -> TransitionBatch:
    # TODO[drjiang]: Will properly handle the None pyre errors in this function,
    # in a subsequent diff. Errors are due to parts of the transition potentially
    # being None.
    state_batch = torch.cat([x.state for x in transitions])
    action_batch = torch.cat([x.action for x in transitions])
    reward_batch = torch.cat([x.reward for x in transitions])
    done_batch = torch.cat([x.done for x in transitions])  # pyre-ignore

    next_state_batch, next_action_batch = None, None
    if has_next_state:
        next_state_batch = torch.cat([x.next_state for x in transitions])  # pyre-ignore
    if has_next_action:
        next_action_batch = torch.cat(
            [x.next_action for x in transitions]  # pyre-ignore
        )

    curr_available_actions_batch, curr_available_actions_mask_batch = None, None
    if not is_action_continuous:
        curr_available_actions_batch = torch.cat(
            [x.curr_available_actions for x in transitions]  # pyre-ignore
        )
        curr_available_actions_mask_batch = torch.cat(
            [x.curr_available_actions_mask for x in transitions]  # pyre-ignore
        )

    next_available_actions_batch, next_available_actions_mask_batch = None, None
    if not is_action_continuous and has_next_available_actions:
        next_available_actions_batch = torch.cat(
            [x.next_available_actions for x in transitions]  # pyre-ignore
        )
        next_available_actions_mask_batch = torch.cat(
            [x.next_available_actions_mask for x in transitions]  # pyre-ignore
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
    )
