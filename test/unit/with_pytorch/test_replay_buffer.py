# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import unittest

import torch

from pearl.policy_learners.sequential_decision_making.ppo import PPOReplayBuffer

from pearl.policy_learners.sequential_decision_making.reinforce import (
    REINFORCEReplayBuffer,
)

from pearl.replay_buffers import BasicReplayBuffer
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import (
    BootstrapReplayBuffer,
)
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class TestReplayBuffer(unittest.TestCase):
    def test_replay_buffer_not_stored_on_gpu(self) -> None:
        replay_buffer_size = 100_000
        number_of_transitions = 80_000
        replay_buffers_to_be_tested = {
            PPOReplayBuffer(replay_buffer_size),
            REINFORCEReplayBuffer(replay_buffer_size),
            REINFORCEReplayBuffer(replay_buffer_size),
            BasicReplayBuffer(replay_buffer_size),
            BootstrapReplayBuffer(replay_buffer_size, p=0.5, ensemble_size=3),
            # We meant to test SARSAReplayBuffer as well, but we observe
            # that it does not get filled when input is random, because
            # it requires the next state of one transition
            # to be equal the current state of the next transition.
            # TODO: verify that we really need this restriction on this
            # replay buffer.
            # SARSAReplayBuffer,
        }
        for replay_buffer in replay_buffers_to_be_tested:
            print(f"Replay buffer class being tested: {type(replay_buffer).__name__}")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            replay_buffer.device_for_batches = device
            action_space = DiscreteActionSpace([torch.tensor([0]), torch.tensor([1])])

            initial_gpu_usage = torch.cuda.memory_allocated(device)
            for _ in range(number_of_transitions):
                replay_buffer.push(
                    state=torch.randn((10, 20)),
                    action=torch.tensor([0]),
                    reward=torch.tensor([4.0]),
                    next_state=torch.randn((10, 20)),
                    curr_available_actions=action_space,
                    next_available_actions=action_space,
                    terminated=(False),
                    truncated=(False),
                    max_number_actions=2,
                )

            # GPU usage must not have been altered because replay buffer is in CPU
            gpu_usage_after_replay_buffer = torch.cuda.memory_allocated(device)
            print(f"Initial GPU usage: {initial_gpu_usage}")
            print(
                f"GPU usage after filling replay buffer: {gpu_usage_after_replay_buffer}"
            )

            # This condition is a bit tricky.
            # Ideally, it should be
            # assertTrue(gpu_usage_after_replay_buffer == initial_gpu_usage)
            # However, we observe that the batch from the previous iteration
            # (see end of this loop) remain in the GPU when initial_gpu_usage
            # is read, and may be freed by the time we are done filling the replay buffer.
            # Therefore, we relax the condition to require that the allocated memory
            # is NOT larger than what we started with (it is ok to be less).
            self.assertFalse(gpu_usage_after_replay_buffer > initial_gpu_usage)

            if torch.cuda.is_available():
                # GPU usage must be increased after sampling batch, which goes to GPU
                _ = replay_buffer.sample(100)
                gpu_usage_after_batch = torch.cuda.memory_allocated(device)
                print(f"GPU usage after sampling batch: {gpu_usage_after_batch}")
                self.assertTrue(gpu_usage_after_batch > gpu_usage_after_replay_buffer)
