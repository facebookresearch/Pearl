# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import unittest

import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.api.action import Action
from pearl.api.state import SubjectiveState

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)

from pearl.replay_buffers.sequential_decision_making.hindsight_experience_replay_buffer import (
    HindsightExperienceReplayBuffer,
)

from pearl.utils.functional_utils.train_and_eval.online_learning import (
    run_episode,
    target_return_is_reached,
)
from pearl.utils.instantiations.environments.sparse_reward_environment import (
    DiscreteSparseRewardEnvironment,
)
from pearl.utils.tensor_like import assert_is_tensor_like


class TestIntegrationReplayBuffer(unittest.TestCase):
    """
    Integration test for replay buffer
    """

    def test_her(self) -> None:
        """
        This test is to ensure HER works for sparse reward environment
                DQN is not able to solve this problem within 1000 episodes
        """
        env: DiscreteSparseRewardEnvironment = DiscreteSparseRewardEnvironment(
            width=50,
            height=50,
            step_size=1,
            action_count=8,
            max_episode_duration=1000,
            reward_distance=1,
        )

        def terminated_fn(state: SubjectiveState, action: Action) -> bool:
            state = assert_is_tensor_like(state)
            next_state = state[:2] + torch.Tensor(env._actions[action]).to(state.device)
            goal = state[-2:]
            if torch.all(torch.eq(next_state, goal)):
                return True
            return False

        def reward_fn(state: SubjectiveState, action: Action) -> int:
            terminated = terminated_fn(state, action)
            if terminated:
                return 0
            return -1

        action_representation_module = OneHotActionTensorRepresentationModule(
            env.action_space.n
        )

        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=4,
                action_space=env.action_space,
                hidden_dims=[64, 64],
                training_rounds=10,
                batch_size=500,
                action_representation_module=action_representation_module,
            ),
            replay_buffer=HindsightExperienceReplayBuffer(
                2_000_000, goal_dim=2, reward_fn=reward_fn, terminated_fn=terminated_fn
            ),
        )
        # precollect data
        for _ in range(500):
            run_episode(
                agent, env, learn=False, exploit=False, learn_after_episode=False
            )

        self.assertTrue(
            target_return_is_reached(
                agent=agent,
                env=env,
                # -600 and 1000 are good enough to differ perf from DQN
                # set it looser for HER to make test faster
                target_return=-600,
                max_episodes=1000,
                learn=True,
                learn_after_episode=True,
                exploit=False,
                check_moving_average=True,
            )
        )
