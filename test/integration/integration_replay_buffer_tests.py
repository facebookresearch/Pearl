#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import unittest

import torch
from pearl.core.sequential_decision_making.policy_learners.deep_q_learning import (
    DeepQLearning,
)

from pearl.pearl_agent import PearlAgent

from pearl.replay_buffers.sequential_decision_making.hindsight_experience_replay_buffer import (
    HindsightExperienceReplayBuffer,
)

from pearl.utils.functional_utils.train_and_eval.online_learning import (
    episode_return,
    target_return_is_reached,
)
from pearl.utils.instantiations.environments.sparse_reward_environment import (
    DiscreteSparseRewardEnvironment,
    SparseRewardEnvSummarizationModule,
)


class IntegrationReplayBufferTests(unittest.TestCase):
    """
    Integration test for replay buffer
    """

    def test_her(self) -> None:
        """
        This test is to ensure HER works for sparse reward environment
                DQN is not able to solve this problem within 1000 episodes
        """
        env = DiscreteSparseRewardEnvironment(
            length=50,
            height=50,
            step_size=1,
            action_count=8,
            max_episode_duration=1000,
            reward_distance=1,
        )

        def done_fn(state, action):
            next_state = state[:2] + torch.Tensor(env._actions[action])
            goal = state[-2:]
            if torch.all(torch.eq(next_state, goal)):
                return True
            return False

        def reward_fn(state, action):
            done = done_fn(state, action)
            if done:
                return 0
            return -1

        agent = PearlAgent(
            policy_learner=DeepQLearning(
                4,
                env.action_space,
                [64, 64],
                training_rounds=10,
                batch_size=500,
            ),
            replay_buffer=HindsightExperienceReplayBuffer(
                2_000_000, goal_dim=2, reward_fn=reward_fn, done_fn=done_fn
            ),
            history_summarization_module=SparseRewardEnvSummarizationModule(),
        )
        # precollect data
        for _ in range(500):
            episode_return(agent, env, False, False, False)

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
