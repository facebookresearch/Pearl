import random
from collections import deque

import torch
import torch.nn.functional as F

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState

from pearl.replay_buffer.replay_buffer import ReplayBuffer
from pearl.replay_buffer.transition import Transition, TransitionBatch
from pearl.replay_buffer.utils import create_next_action_tensor_and_mask


class FIFOOffPolicyReplayBuffer(ReplayBuffer):
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
        next_available_actions: ActionSpace,
        action_space: ActionSpace,
        done: bool,
    ) -> None:
        state_tensor = torch.tensor(state).unsqueeze(0)  # (1 x state_dim)
        action_tensor = F.one_hot(
            torch.tensor([action]), num_classes=action_space.n
        )  # (1 x action_dim)
        reward_tensor = torch.tensor([reward])  # (1)
        next_state_tensor = torch.tensor(next_state).unsqueeze(0)  # (1 x state_dim)

        (
            next_available_actions_tensor_with_padding,
            next_available_actions_mask,
        ) = create_next_action_tensor_and_mask(action_space, next_available_actions)

        done_tensor = torch.tensor([done]).float()  # (1)

        self.memory.append(
            Transition(
                state=state_tensor,
                action=action_tensor,
                reward=reward_tensor,
                next_state=next_state_tensor,
                next_available_actions=next_available_actions_tensor_with_padding,
                next_available_actions_mask=next_available_actions_mask,
                done=done_tensor,
            )
        )

    # input: batch_size
    # output: TransitionBatch(
    #   state = tensor(batch_size, state_dim),
    #   action = tensor(batch_size, action_dim),
    #   reward = tensor(batch_size, ),
    #   next_state = tensor(batch_size, state_dim),
    #   next_available_actions = tensor(batch_size, action_dim, action_dim),
    #   next_available_actions_mask = tensor(batch_size, action_dim),
    #   done = tensor(batch_size, ),
    # )
    def sample(self, batch_size: int) -> TransitionBatch:
        samples = random.sample(self.memory, batch_size)
        return TransitionBatch(
            state=torch.cat([x.state for x in samples]),
            action=torch.cat([x.action for x in samples]),
            reward=torch.cat([x.reward for x in samples]),
            next_state=torch.cat([x.next_state for x in samples]),
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
