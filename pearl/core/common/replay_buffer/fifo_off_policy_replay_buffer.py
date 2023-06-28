import random
from collections import deque

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState

from pearl.core.common.replay_buffer.tensor_based_replay_buffer import (
    TensorBasedReplayBuffer,
)
from pearl.core.common.replay_buffer.transition import Transition, TransitionBatch


class FIFOOffPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    # TODO: add helper to convert subjective state into tensors
    # TODO: assumes action space is gym action space with one-hot encoding
    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: float,
        next_state: SubjectiveState,
        curr_available_actions: ActionSpace,
        next_available_actions: ActionSpace,
        action_space: ActionSpace,
        done: bool,
    ) -> None:

        (
            curr_available_actions_tensor_with_padding,
            curr_available_actions_mask,
        ) = TensorBasedReplayBuffer._create_action_tensor_and_mask(
            action_space, curr_available_actions
        )

        (
            next_available_actions_tensor_with_padding,
            next_available_actions_mask,
        ) = TensorBasedReplayBuffer._create_action_tensor_and_mask(
            action_space, next_available_actions
        )

        self.memory.append(
            Transition(
                state=TensorBasedReplayBuffer._process_single_state(state),
                action=TensorBasedReplayBuffer._process_single_action(
                    action, action_space
                ),
                reward=TensorBasedReplayBuffer._process_single_reward(reward),
                next_state=TensorBasedReplayBuffer._process_single_state(next_state),
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_available_actions_mask=curr_available_actions_mask,
                next_available_actions=next_available_actions_tensor_with_padding,
                next_available_actions_mask=next_available_actions_mask,
                done=TensorBasedReplayBuffer._process_single_done(done),
            )
        )

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
        samples = random.sample(self.memory, batch_size)
        return TransitionBatch(
            state=torch.cat([x.state for x in samples]),
            action=torch.cat([x.action for x in samples]),
            reward=torch.cat([x.reward for x in samples]),
            next_state=torch.cat([x.next_state for x in samples]),
            curr_available_actions=torch.cat(
                [x.curr_available_actions for x in samples]
            ),
            curr_available_actions_mask=torch.cat(
                [x.curr_available_actions_mask for x in samples]
            ),
            next_available_actions=torch.cat(
                [x.next_available_actions for x in samples]
            ),
            next_available_actions_mask=torch.cat(
                [x.next_available_actions_mask for x in samples]
            ),
            done=torch.cat([x.done for x in samples]),
        )

    def __len__(self) -> int:
        return len(self.memory)
