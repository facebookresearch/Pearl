# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import os
import random
import unittest

import numpy as np

import torch
import torch.nn as nn
from pearl.action_representation_modules.identity_action_representation_module import (
    IdentityActionRepresentationModule,
)
from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.api.space import Space
from pearl.history_summarization_modules.lstm_history_summarization_module import (
    LSTMHistorySummarizationModule,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    EnsembleQValueNetwork,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.bootstrapped_dqn import (
    BootstrappedDQN,
)
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.replay_buffers import BasicReplayBuffer
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import (
    BootstrapReplayBuffer,
)
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.utils.instantiations.spaces.box import BoxSpace
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1

"""
This is a unit test version of the Recommender System tutorial.
It is meant to check whether code changes break the tutorial.
It is therefore important that the tutorial and the code here are kept in sync.
As part of that synchronization, the markdown cells in the tutorial are
kept here as multi-line strings.

For it to run quickly, the number of steps used for training is reduced.
"""


number_of_steps = 30  # 100000


"""
## Load Environment
This environment's underlying model was pre-trained using the MIND dataset (Wu et al. 2020).
The model is defined by class `SequenceClassificationModel` below.
The model's state dict is saved in
tutorials/single_item_recommender_system_example/env_model_state_dict.pt

Each data point is:
- A history of impressions clicked by a user
- Each impression is represented by an 100-dim vector
- A list of impressions and whether or not they are clicked

The environment is constructed with the setup below. Note that this is a contrived example
to illustrate Pearl's usage, agent modularity and a subset of features,
not to represent a real-world environment or problem.
- State: a history of impressions by a user (note that we used the history of impressions
  instead of clicked impressions to speed up learning in this example).
  Interested Pearl users can change it to history of clicked impressions with much longer
  episode length and samples to run the following experiments.)
- Dynamic action space: two randomly picked impressions
- Action: one of the two impressions
- Reward: click
- Reset every 20 steps.
"""


class SequenceClassificationModel(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 128,
        state_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            num_layers=num_layers,
            input_size=observation_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.register_buffer(
            "default_cell_representation", torch.zeros((num_layers, hidden_dim))
        )
        self.register_buffer(
            "default_hidden_representation", torch.zeros((num_layers, hidden_dim))
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        h0 = (
            self.default_hidden_representation.unsqueeze(1)
            .repeat(1, batch_size, 1)
            .detach()
        )
        c0 = (
            self.default_cell_representation.unsqueeze(1)
            .repeat(1, batch_size, 1)
            .detach()
        )
        out, (_, _) = self.lstm(x, (h0, c0))
        mlp_input = out[:, -1, :].view((batch_size, -1))
        return torch.sigmoid(self.mlp(torch.cat([mlp_input, action], dim=-1)))


class RecEnv(Environment):
    def __init__(
        self, actions: list[torch.Tensor], model: nn.Module, history_length: int
    ) -> None:
        self.model: nn.Module = model.to(device)
        self.history_length: int = history_length
        self.t = 0
        self.T = 20
        self.actions: list[list[torch.Tensor]] = [
            [torch.tensor(k) for k in random.sample(actions, 2)] for _ in range(self.T)
        ]
        self.state: torch.Tensor = torch.zeros((self.history_length, 100)).to(device)
        self._action_space: DiscreteActionSpace = DiscreteActionSpace(self.actions[0])

    @property
    def action_space(self) -> ActionSpace:
        return DiscreteActionSpace(self.actions[0])

    @property
    def observation_space(self) -> Space:
        return BoxSpace(low=torch.zeros((1,)), high=torch.ones((1,)))

    def reset(self, seed: int | None = None) -> tuple[Observation, ActionSpace]:
        self.state: torch.Tensor = torch.zeros((self.history_length, 100))
        self.t = 0
        self._action_space: DiscreteActionSpace = DiscreteActionSpace(
            self.actions[self.t]
        )
        return [0.0], self._action_space

    def step(self, action: Action) -> ActionResult:
        action = action.to(device)
        action_batch = action.unsqueeze(0)
        state_batch = self.state.unsqueeze(0).to(device)
        reward = self.model(state_batch, action_batch) * 3  # To speed up learning
        true_reward = np.random.binomial(1, reward.item())
        self.state = torch.cat([self.state[1:, :].to(device), action_batch], dim=0)

        self.t += 1
        if self.t < self.T:
            self._action_space = DiscreteActionSpace(self.actions[self.t])
        return ActionResult(
            observation=[float(true_reward)],
            reward=float(true_reward),
            terminated=self.t >= self.T,
            truncated=False,
            info={},
            available_action_space=self._action_space,
        )

    def __str__(self) -> str:
        return self.__class__.__name__


class TestTutorials(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_rec_system(self) -> None:
        # load environment
        model = SequenceClassificationModel(100).to(device)
        if os.path.exists("../Pearl"):
            # Github CI
            model_dir = "tutorials/single_item_recommender_system_example/"
        else:
            # internal Meta tests
            model_dir = "pearl/tutorials/single_item_recommender_system_example/"

        model.load_state_dict(
            # Note: in the tutorial the directory "pearl" must be replaced by "Pearl"
            torch.load(
                os.path.join(model_dir, "env_model_state_dict.pt"),
                weights_only=True,
            )
        )
        # Note: in the tutorial the directory "pearl" must be replaced by "Pearl"
        actions = torch.load(
            os.path.join(model_dir, "news_embedding_small.pt"),
            weights_only=True,
        )
        history_length = 8
        env = RecEnv(list(actions.values())[:100], model, history_length)
        observation, action_space = env.reset()
        assert isinstance(action_space, DiscreteActionSpace)

        # experiment code
        record_period = 400

        """
        ## Vanilla DQN Agent
        Able to handle dynamic action space but not able to handle partial observability
        and sparse reward.
        """

        # create a pearl agent

        action_representation_module = IdentityActionRepresentationModule(
            max_number_actions=action_space.n,
            representation_dim=action_space.action_dim,
        )

        # DQN-vanilla
        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=1,
                action_space=action_space,
                hidden_dims=[64, 64],
                training_rounds=50,
                action_representation_module=action_representation_module,
            ),
            replay_buffer=BasicReplayBuffer(100_000),
            device_id=device_id,
        )

        # info = \
        online_learning(
            agent=agent,
            env=env,
            number_of_steps=number_of_steps,
            print_every_x_steps=100,
            record_period=min(record_period, number_of_steps),
            learn_after_episode=True,
        )

        # Keep the commented out code below as a reference for the notebook
        # torch.save(info["return"], "DQN-return.pt")
        # plt.plot(
        #     record_period * np.arange(len(info["return"])), info["return"], label="DQN"
        # )
        # plt.legend()
        # plt.show()

        """
        ## DQN Agent with LSTM history summarization module

        Now the DQN agent can handle partially observable environments with history summarization.
        """

        # Add a LSTM history summarization module

        agent = PearlAgent(
            policy_learner=DeepQLearning(
                state_dim=128,
                action_space=action_space,
                hidden_dims=[64, 64],
                training_rounds=50,
                action_representation_module=action_representation_module,
            ),
            history_summarization_module=LSTMHistorySummarizationModule(
                observation_dim=1,
                action_dim=100,
                hidden_dim=128,
                history_length=history_length,
            ),
            replay_buffer=BasicReplayBuffer(100_000),
            device_id=device_id,
        )

        # info = \
        online_learning(
            agent=agent,
            env=env,
            number_of_steps=number_of_steps,
            print_every_x_steps=100,
            record_period=min(record_period, number_of_steps),
            learn_after_episode=True,
        )

        # Keep the commented out code below as a reference for the notebook
        # torch.save(info["return"], "DQN-LSTM-return.pt")
        # plt.plot(
        #     record_period * np.arange(len(info["return"])),
        #     info["return"],
        #     label="DQN-LSTM",
        # )
        # plt.legend()
        # plt.show()

        """
        ## Bootstrapped DQN Agent with LSTM History Summarization

        Leveraging the deep exploration value-based algorithm, now the agent can achieve a
        better performance in a much faster way while being able to still leverage
        history summarization capability.
        Note how top average performance takes around 20,000 steps in the graph above,
        but only about 5,000 steps in the graph below.
        """

        # Better exploration with BootstrappedDQN-LSTM

        agent = PearlAgent(
            policy_learner=BootstrappedDQN(
                q_ensemble_network=EnsembleQValueNetwork(
                    state_dim=128,
                    action_dim=100,
                    ensemble_size=10,
                    output_dim=1,
                    hidden_dims=[64, 64],
                    prior_scale=0.3,
                ),
                action_space=action_space,
                training_rounds=50,
                action_representation_module=action_representation_module,
            ),
            history_summarization_module=LSTMHistorySummarizationModule(
                observation_dim=1,
                action_dim=100,
                hidden_dim=128,
                history_length=history_length,
            ),
            replay_buffer=BootstrapReplayBuffer(100_000, 1.0, 10),
            device_id=device_id,
        )

        # info = \
        online_learning(
            agent=agent,
            env=env,
            number_of_steps=number_of_steps,
            print_every_x_steps=100,
            record_period=min(record_period, number_of_steps),
            learn_after_episode=True,
        )

        # Keep the commented out code below as a reference for the notebook
        # torch.save(info["return"], "BootstrappedDQN-LSTM-return.pt")
        # plt.plot(
        #     record_period * np.arange(len(info["return"])),
        #     info["return"],
        #     label="BootstrappedDQN-LSTM",
        # )
        # plt.legend()
        # plt.show()

        """
        ## Summary
        In this example, we illustrated Pearl's capability of dealing with dynamic action space,
        standard policy learning, history summarization and intelligent exploration,
        all in a single agent.
        """
