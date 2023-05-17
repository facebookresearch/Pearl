from abc import abstractmethod
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.factory import network_maker, NetworkType
from pearl.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffer.replay_buffer import ReplayBuffer
from pearl.replay_buffer.transition import TransitionBatch
from torch import optim


# TODO: Only support discrete action space problems for now and assumes Gym action space.
# Currently available actions is not used. Needs to be updated once we know the input structure
# of production stack on this param.
# TODO: Add a flag for on-policy vs off-policy to distinguish Q learning vs SARSA
class DeepTDLearning(PolicyLearner):
    """
    An Abstract Class for Deep Temporal Difference learning policy learner.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_dims: Iterable[int],
        exploration_module: ExplorationModule,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        capacity: int = 10000,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        network_type: NetworkType = NetworkType.VANILLA,
    ) -> None:
        super(DeepTDLearning, self).__init__(
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        self._action_space = action_space
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._target_update_freq = target_update_freq

        # TODO: Assumes Gym interface, fix it.
        def make_specified_network():
            return network_maker(
                state_dim=state_dim,
                action_dim=action_space.n,
                hidden_dims=hidden_dims,
                output_dim=1,
                network_type=network_type,
            )

        self._Q = make_specified_network()
        self._Q_target = make_specified_network()

        self._Q_target.load_state_dict(self._Q.state_dict())
        self._optimizer = optim.AdamW(
            self._Q.parameters(), lr=learning_rate, amsgrad=True
        )

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def exploit(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
    ) -> Action:
        # TODO: Assumes subjective state is a torch tensor and gym action space.
        # Fix the available action space.
        with torch.no_grad():
            subjective_state_tensor = torch.tensor(subjective_state)  # (state_dim)
            states_repeated = torch.repeat_interleave(
                subjective_state_tensor.unsqueeze(0), available_action_space.n, dim=0
            )  # (action_space_size x state_dim)
            actions = F.one_hot(
                torch.arange(0, available_action_space.n)
            )  # (action_space_size, action_dim)
            state_action_pairs = torch.cat(
                [states_repeated, actions], dim=1
            )  # (action_space_size x (state_dim + action_dim))
            exploit_action = torch.argmax(self._Q(state_action_pairs)).view((-1)).item()

        return exploit_action

    @abstractmethod
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.tensor:
        pass

    def learn_batch(self, batch: TransitionBatch) -> None:
        state_batch = batch.state  # (batch_size x state_dim)
        action_batch = batch.action  # (batch_size x action_dim)
        reward_batch = batch.reward  # (batch_size)
        done_batch = batch.done  # (batch_size)
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_available_actions_batch = (
            batch.next_available_actions
        )  # (batch_size x action_space_size x action_dim)
        next_available_actions_mask_batch = (
            batch.next_available_actions_mask
        )  # (batch_size x action_space_size)

        batch_size = state_batch.shape[0]
        # sanity check they have same batch_size
        assert action_batch.shape[0] == batch_size
        assert reward_batch.shape[0] == batch_size
        assert done_batch.shape[0] == batch_size
        assert next_state_batch.shape[0] == batch_size
        assert next_available_actions_batch.shape[0] == batch_size
        assert next_available_actions_mask_batch.shape[0] == batch_size

        state_action_values = self._Q(
            torch.cat([state_batch, action_batch], dim=1)
        ).view(
            (-1)
        )  # (batch_size)

        # Compute the Bellman Target
        expected_state_action_values = (
            self._get_next_state_values(batch, batch_size)
            * self._discount_factor
            * (1 - done_batch)
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        criterion = torch.nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Target Network Update
        if (self._training_steps + 1) % self._target_update_freq == 0:
            self._Q_target.load_state_dict(self._Q.state_dict())
