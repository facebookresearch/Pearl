from pearl.neural_networks.common.value_networks import EnsembleQValueNetwork
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import BootstrapReplayBuffer
from pearl.policy_learners.sequential_decision_making.bootstrapped_dqn import BootstrappedDQN
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.policy_learners.sequential_decision_making.deep_q_learning import DeepQLearning
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.pearl_agent import PearlAgent
from pearl.tutorials.single_item_recommender_system_example.env_model import SequenceClassificationModel
from pearl.tutorials.single_item_recommender_system_example.env import RecEnv
import torch
import matplotlib.pyplot as plt
import numpy as np
set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SequenceClassificationModel(100).to(device)
model.load_state_dict(torch.load("/home/lz/PycharmProjects/Pearl/pearl/tutorials/single_item_recommender_system_example/env_model_state_dict.pt"))
actions = torch.load("/home/lz/PycharmProjects/Pearl/pearl/tutorials/single_item_recommender_system_example/news_embedding_small.pt")
env = RecEnv(list(actions.values())[:100], model)
observation, action_space = env.reset()

# experiment code
number_of_steps = 100000
record_period = 400



# create a pearl agent

action_representation_module = IdentityActionRepresentationModule(
    max_number_actions=action_space.n,
    representation_dim=action_space.action_dim,
)

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
        history_length=8,
    ),
    replay_buffer=BootstrapReplayBuffer(100_000, 1.0, 10),
    device_id=-1,
)

info = online_learning(
    agent=agent,
    env=env,
    number_of_steps=number_of_steps,
    print_every_x_steps=100,
    record_period=record_period,
    learn_after_episode=True,
)
torch.save(info["return"], "BootstrappedDQN-LSTM-return.pt")
plt.plot(record_period * np.arange(len(info["return"])), info["return"], label="BootstrappedDQN-LSTM")
plt.legend()
plt.show()
