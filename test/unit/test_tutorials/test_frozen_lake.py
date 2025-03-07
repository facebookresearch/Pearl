# pyre-unsafe
import unittest

# import matplotlib.pyplot as plt
# import numpy as np
import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.environments.environments import (
    OneHotObservationsFromDiscrete,
)
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1

"""
This is a unit test version of the frozen lake example tutorial.
It is meant to check whether code changes break the tutorial.
It is therefore important that the tutorial and the code here are kept in sync.
As part of that synchronization, the markdown cells in the tutorial are
kept here as multi-line strings.


For it to run quickly, the number of steps used for training is reduced.
"""


class TestFrozenLakeTutorial(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.env = OneHotObservationsFromDiscrete(
            GymEnvironment(
                "FrozenLake-v1",
                is_slippery=False,
                map_name="4x4",
            )
        )
        assert isinstance(self.env.observation_space, DiscreteSpace)
        self.state_dim = self.env.observation_space.n
        self.number_of_steps = 2000
        self.record_period = 400

    # Step 1: Instantiate a DQN agent
    def instantiate_dqn_agent(self) -> PearlAgent:
        # We use a one hot representation for representing actions. Take action_dim = num_actions.
        assert isinstance(self.env.action_space, DiscreteSpace)
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=self.env.action_space.n,
        )

        DQNagent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=self.state_dim,
                action_space=self.env.action_space,
                hidden_dims=[64, 64],
                training_rounds=1,
                action_representation_module=action_representation_module,
            ),
            replay_buffer=BasicReplayBuffer(1000),
        )

        return DQNagent

    # Step 3: Train the agent: online interaction with the environment and learning
    def test_dqn_frozen_lake(self) -> None:
        DQNagent = self.instantiate_dqn_agent()

        # Online learning function implements environment interaction and learning and
        # returns a dictionary with episodic returns
        info = online_learning(  # noqa
            agent=DQNagent,
            env=self.env,
            number_of_steps=self.number_of_steps,
            print_every_x_steps=100,  # print episodic returns and steps after every 100 episodes
            record_period=self.record_period,
            learn_after_episode=False,  # Q-network update after every environment step  # noqa
        )

        # Keep the commented out code below as a reference for the notebook
        # torch.save(info["return"], "DQN-return.pt")
        # plt.plot(
        #     self.record_period * np.arange(len(info["return"])),
        #     info["return"],
        #     label="DQN",
        # )
        # plt.xlabel("steps")
        # plt.ylabel("return")
        # plt.legend()
        # plt.show()
