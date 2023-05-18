import random
from collections import deque

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.replay_buffer.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffer.transition import Transition, TransitionBatch


class FIFOOnPolicyReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        # this is used to delay push SARS
        # wait for next action is available and then final push
        # this is designed for single transition for now
        self.cache = None

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
        (
            next_available_actions_tensor_with_padding,
            next_available_actions_mask,
        ) = TensorBasedReplayBuffer._create_next_action_tensor_and_mask(
            action_space, next_available_actions
        )
        current_state = TensorBasedReplayBuffer._process_single_state(state)
        current_action = TensorBasedReplayBuffer._process_single_action(
            action, action_space
        )

        find_match = self.cache is not None and torch.equal(
            self.cache.next_state, current_state
        )
        if find_match:
            # push a complete SARSA into memory
            self.memory.append(
                Transition(
                    state=self.cache.state,
                    action=self.cache.action,
                    reward=self.cache.reward,
                    next_state=self.cache.next_state,
                    next_action=current_action,
                    next_available_actions=self.cache.next_available_actions,
                    next_available_actions_mask=self.cache.next_available_actions_mask,
                    done=self.cache.done,
                )
            )
        # save current push into cache
        self.cache = Transition(
            state=current_state,
            action=current_action,
            reward=TensorBasedReplayBuffer._process_single_reward(reward),
            next_state=TensorBasedReplayBuffer._process_single_state(next_state),
            next_available_actions=next_available_actions_tensor_with_padding,
            next_available_actions_mask=next_available_actions_mask,
            done=TensorBasedReplayBuffer._process_single_done(done),
        )

    def sample(self, batch_size: int) -> TransitionBatch:
        samples = random.sample(self.memory, batch_size)
        return TransitionBatch(
            state=torch.cat([x.state for x in samples]),
            action=torch.cat([x.action for x in samples]),
            reward=torch.cat([x.reward for x in samples]),
            next_state=torch.cat([x.next_state for x in samples]),
            next_action=torch.cat([x.next_action for x in samples]),
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
