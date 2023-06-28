from abc import abstractmethod
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.core.common.neural_networks.value_networks import (
    StateActionValueNetworkType,
    TwoTowerStateActionValueNetwork,
    VanillaStateActionValueNetwork,
)
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.core.common.policy_learners.policy_learner import PolicyLearner
from pearl.core.common.replay_buffer.transition import TransitionBatch
from torch import optim


# TODO: Only support discrete action space problems for now and assumes Gym action space.
# Currently available actions is not used. Needs to be updated once we know the input structure
# of production stack on this param.
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
        on_policy: bool,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        capacity: int = 10000,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 0.1,
        network_type: StateActionValueNetworkType = VanillaStateActionValueNetwork,
        state_output_dim=None,
        action_output_dim=None,
        state_hidden_dims=None,
        action_hidden_dims=None,
    ) -> None:
        super(DeepTDLearning, self).__init__(
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            on_policy=on_policy,
        )
        self._action_space = action_space
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._target_update_freq = target_update_freq
        self._soft_update_tau = soft_update_tau

        # TODO: Assumes Gym interface, fix it.
        def make_specified_network():
            if network_type == TwoTowerStateActionValueNetwork:
                return network_type(
                    state_dim=state_dim,
                    action_dim=action_space.n,
                    hidden_dims=hidden_dims,
                    state_output_dim=state_output_dim,
                    action_output_dim=action_output_dim,
                    state_hidden_dims=state_hidden_dims,
                    action_hidden_dims=action_hidden_dims,
                    output_dim=1,
                )
            else:
                return network_type(
                    state_dim=state_dim,
                    action_dim=action_space.n,
                    hidden_dims=hidden_dims,
                    output_dim=1,
                )

        self._Q = make_specified_network()
        self._Q_target = make_specified_network()

        self._Q_target.load_state_dict(self._Q.state_dict())
        self._optimizer = optim.AdamW(
            self._Q.parameters(), lr=learning_rate, amsgrad=True
        )

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
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
            q_values = self._Q(state_action_pairs)  # (action_space_size, 1)
            exploit_action = torch.argmax(q_values).view((-1)).item()

        if exploit:
            return exploit_action

        return self._exploration_module.act(
            subjective_state,
            available_action_space,
            exploit_action,
            q_values,
        )

    @abstractmethod
    def _get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.tensor:
        pass

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
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

        state_action_values = self._Q.get_batch_action_value(
            state_batch=state_batch,
            action_batch=action_batch,
            curr_available_actions_batch=batch.curr_available_actions,
        )

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
            self._update_target_network()

        return {
            "loss": torch.abs(state_action_values - expected_state_action_values)
            .mean()
            .item()
        }

    def _update_target_network(self):
        # Q_target = tao * Q_target + (1-tao)*Q
        target_net_state_dict = self._Q_target.state_dict()
        source_net_state_dict = self._Q.state_dict()
        for key in source_net_state_dict:
            target_net_state_dict[key] = (
                self._soft_update_tau * source_net_state_dict[key]
                + (1 - self._soft_update_tau) * target_net_state_dict[key]
            )

        self._Q_target.load_state_dict(target_net_state_dict)
