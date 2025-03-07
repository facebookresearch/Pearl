# pyre-unsafe
import unittest

# import matplotlib.pyplot as plt
import torch
from pearl.action_representation_modules.one_hot_action_representation_module import (
    OneHotActionTensorRepresentationModule,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.double_dqn import DoubleDQN
from pearl.replay_buffers import BasicReplayBuffer

from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1

"""
This is a unit test version of the DQN and Double DQN example tutorial.
It is meant to check whether code changes break the tutorial.
It is therefore important that the tutorial and the code here are kept in sync.
As part of that synchronization, the markdown cells in the tutorial are
kept here as multi-line strings.


For it to run quickly, the number of steps used for training is reduced.
"""


class TestDQNTutorial(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.env = GymEnvironment("CartPole-v1")
        assert isinstance(self.env.action_space, DiscreteActionSpace)
        self.num_actions = self.env.action_space.n

    # Step 1: Instantiate a 2 layer Q value network
    def instantiate_q_network(self) -> QValueNetwork:
        hidden_dims = [64, 64]

        # Note: VanillaQValueNetwork class uses a simple mlp for approximating the Q values.
        #  - Input dimension of the mlp = (state_dim + action_dim)
        #  - Size of the intermediate layers are specified as list of `hidden_dims`.
        #  - We use a one hot representation for representing actions.
        #    Take action_dim = num_actions.

        Q_value_network = VanillaQValueNetwork(
            state_dim=self.env.observation_space.shape[
                0
            ],  # state representation dimension
            action_dim=self.num_actions,  # action representation dimension
            hidden_dims=hidden_dims,  # hidden layers
            output_dim=1,  # set to 1 (Q values are scalars)
        )

        return Q_value_network

    # Step 2: Instantiate a DQN agent
    def instantiate_dqn_agent(self) -> PearlAgent:
        # We use a one hot representation for representing actions. Take action_dim = num_actions.
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=self.num_actions
        )

        Q_value_network = self.instantiate_q_network()

        # Note: Instead of using the 'network_type' argument, use the 'network_instance' argument
        # when passing Q_value_network to the `DeepQLearning` policy learner.
        #  - This allows for custom implementations of Q-value networks.
        #  - 'network_type' argument will be deprecated in the future
        DQNagent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=self.env.observation_space.shape[0],
                action_space=self.env.action_space,
                batch_size=64,
                training_rounds=10,
                soft_update_tau=0.75,
                network_instance=Q_value_network,  # Q_value_network instance to policy learner
                action_representation_module=action_representation_module,
            ),
            replay_buffer=BasicReplayBuffer(10_000),
        )

        return DQNagent

    # Step 3: Train the agent: online interaction with the environment and learning
    def test_dqn(self) -> None:
        DQNagent = self.instantiate_dqn_agent()
        # Online learning function implements environment interaction and learning and
        # returns a dictionary with episodic returns
        info = online_learning(  # noqa
            agent=DQNagent,
            env=self.env,
            number_of_episodes=40,
            print_every_x_episodes=10,  # print episodic returns and steps after every 20 episodes
            learn_after_episode=True,  # Q-network update at the end of each episode instead of every environment step  # noqa
            seed=0,
        )

        # Keep the commented out code below as a reference for the notebook
        # torch.save(
        #     info["return"], "DQN-return.pt"
        # )  # info["return"] refers to the episodic returns
        # plt.plot(np.arange(len(info["return"])), info["return"], label="DQN")
        # plt.title("Episodic returns")
        # plt.xlabel("Episode")
        # plt.ylabel("Return")
        # plt.legend()
        # plt.show()


class TestDoubleDQNTutorial(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.env = GymEnvironment("CartPole-v1")
        assert isinstance(self.env.action_space, DiscreteActionSpace)
        self.num_actions = self.env.action_space.n

    # Step 1: Instantiate the environment and a 2 layer Q value network
    def instantiate_q_network(self) -> QValueNetwork:
        hidden_dims = [64, 64]

        # We use a one hot representation for representing actions. Take action_dim = num_actions.
        Q_value_network = VanillaQValueNetwork(
            state_dim=self.env.observation_space.shape[
                0
            ],  # state representation dimension
            action_dim=self.num_actions,  # action representation dimension
            hidden_dims=hidden_dims,  # hidden layers
            output_dim=1,  # set to 1 (Q values are scalars)
        )

        return Q_value_network

    # Step 2: Instantiate a Double DQN agent
    def instantiate_doubledqn_agent(self) -> PearlAgent:
        # We use a one hot representation for representing actions. Take action_dim = num_actions.
        action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=self.num_actions
        )

        Q_value_network = self.instantiate_q_network()

        DoubleDQNagent = PearlAgent(
            policy_learner=DoubleDQN(
                state_dim=self.env.observation_space.shape[0],
                action_space=self.env.action_space,
                batch_size=64,
                training_rounds=10,
                soft_update_tau=0.75,
                network_instance=Q_value_network,  # Q_value_network instance passed to policy learner  # noqa
                action_representation_module=action_representation_module,
            ),
            replay_buffer=BasicReplayBuffer(10_000),
        )

        return DoubleDQNagent

    # Step 3: Train the agent: online interaction with the environment and learning

    # Online learning function implements environment interaction and learning and
    # returns a dictionary with episodic returns
    def test_double_dqn(self) -> None:
        DoubleDQNagent = self.instantiate_doubledqn_agent()
        info_DoubleDQN = online_learning(  # noqa
            agent=DoubleDQNagent,
            env=self.env,
            number_of_episodes=40,
            print_every_x_episodes=10,  # print returns after every 10 episdoes
            learn_after_episode=True,  # Q-network update at the end of each episode instead of every environment step  # noqa
            seed=0,
        )

        # Keep the commented out code below as a reference for the notebook
        # torch.save(
        #     info_DoubleDQN["return"], "DoubleDQN-return.pt"
        # )  # info["return"] refers to the episodic returns
        # plt.plot(
        #     np.arange(len(info_DoubleDQN["return"])),
        #     info_DoubleDQN["return"],
        #     label="DoubleDQN",
        # )
        # plt.title("Episodic returns")
        # plt.xlabel("Episode")
        # plt.ylabel("Return")
        # plt.legend()
        # plt.show()
