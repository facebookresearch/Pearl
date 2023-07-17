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


class OnPolicyEpisodicReplayBuffer(TensorBasedReplayBuffer):
    def __init__(self, capacity: int, discounted_factor: float = 1.0) -> None:
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        # this is used to delay push SARS
        # wait for next action is available and then final push
        # this is designed for single transition for now
        self.reward_cache = []
        self.state_action_cache = []
        self._discounted_factor = discounted_factor

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

        current_state = TensorBasedReplayBuffer._process_single_state(state)
        current_action = TensorBasedReplayBuffer._process_single_action(
            action, action_space
        )

        self.reward_cache.append(reward)
        self.state_action_cache.append(
            Transition(
                state=current_state,
                action=current_action,
                reward=None,
                next_state=None,
                curr_available_actions=curr_available_actions_tensor_with_padding,
                curr_available_actions_mask=curr_available_actions_mask,
                next_available_actions=None,
                next_available_actions_mask=None,
                done=TensorBasedReplayBuffer._process_single_done(done),
            )
        )

        if done:
            # discounted_return at time i = sum of (self._discounted_factor^(j-i) * Rj) j is [i, T]
            discounted_return = 0
            for i in range(len(self.state_action_cache) - 1, -1, -1):
                cum_return = self.reward_cache[i] + discounted_return
                self.state_action_cache[
                    i
                ].reward = TensorBasedReplayBuffer._process_single_reward(cum_return)
                self.memory.append(self.state_action_cache[i])
                discounted_return = self._discounted_factor * cum_return

            self.reward_cache = []
            self.state_action_cache = []

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size > len(self):
            raise ValueError(
                f"Can't get a batch of size {batch_size} from a replay buffer with only {len(self)} elements"
            )
        samples = random.sample(self.memory, batch_size)
        return TransitionBatch(
            state=torch.cat([x.state for x in samples]),
            action=torch.cat([x.action for x in samples]),
            reward=torch.cat([x.reward for x in samples]),
            next_state=None,
            next_action=None,
            curr_available_actions=torch.cat(
                [x.curr_available_actions for x in samples]
            ),
            curr_available_actions_mask=torch.cat(
                [x.curr_available_actions_mask for x in samples]
            ),
            next_available_actions=None,
            next_available_actions_mask=None,
            done=torch.cat([x.done for x in samples]),
        )

    def empty(self) -> None:
        self.memory = deque([], maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.memory)
