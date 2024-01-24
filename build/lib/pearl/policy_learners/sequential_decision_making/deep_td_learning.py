# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.neural_networks.common.utils import update_target_network

from pearl.neural_networks.common.value_networks import (
    DuelingQValueNetwork,
    TwoTowerQValueNetwork,
    VanillaQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch

from pearl.utils.functional_utils.learning.loss_fn_utils import compute_cql_loss
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
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
        exploration_module: ExplorationModule,
        on_policy: bool,
        action_space: Optional[ActionSpace] = None,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 0.1,
        is_conservative: bool = False,
        conservative_alpha: float = 2.0,
        network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        state_output_dim: Optional[int] = None,
        action_output_dim: Optional[int] = None,
        state_hidden_dims: Optional[List[int]] = None,
        action_hidden_dims: Optional[List[int]] = None,
        network_instance: Optional[QValueNetwork] = None,
        action_representation_module: Optional[ActionRepresentationModule] = None,
        **kwargs: Any,
    ) -> None:
        super(DeepTDLearning, self).__init__(
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            on_policy=on_policy,
            is_action_continuous=False,
            action_representation_module=action_representation_module,
            action_space=action_space,
        )
        self._action_space = action_space
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._target_update_freq = target_update_freq
        self._soft_update_tau = soft_update_tau
        self._is_conservative = is_conservative
        self._conservative_alpha = conservative_alpha

        def make_specified_network() -> QValueNetwork:
            assert hidden_dims is not None
            if network_type is TwoTowerQValueNetwork:
                return network_type(
                    state_dim=state_dim,
                    action_dim=self._action_representation_module.representation_dim,
                    hidden_dims=hidden_dims,
                    state_output_dim=state_output_dim,
                    action_output_dim=action_output_dim,
                    state_hidden_dims=state_hidden_dims,
                    action_hidden_dims=action_hidden_dims,
                    output_dim=1,
                )
            else:
                assert (
                    network_type is VanillaQValueNetwork
                    or network_type is DuelingQValueNetwork
                )
                return network_type(
                    state_dim=state_dim,
                    action_dim=self._action_representation_module.representation_dim,
                    hidden_dims=hidden_dims,
                    output_dim=1,
                )

        if network_instance is not None:
            self._Q: QValueNetwork = network_instance
        else:
            self._Q = make_specified_network()

        self._Q_target: QValueNetwork = copy.deepcopy(self._Q)
        self._optimizer: torch.optim.Optimizer = optim.AdamW(
            self._Q.parameters(), lr=learning_rate, amsgrad=True
        )

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        self._optimizer.add_param_group({"params": value.parameters()})
        self._history_summarization_module = value

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        # TODO: Assumes gym action space.
        # Fix the available action space.
        assert isinstance(available_action_space, DiscreteActionSpace)
        with torch.no_grad():
            states_repeated = torch.repeat_interleave(
                subjective_state.unsqueeze(0),
                available_action_space.n,
                dim=0,
            )
            # (action_space_size x state_dim)

            actions = self._action_representation_module(
                available_action_space.actions_batch.to(states_repeated)
            )
            # (action_space_size, action_dim)

            q_values = self._Q.get_q_values(states_repeated, actions)
            # this does a forward pass since all avaialble
            # actions are already stacked together

            exploit_action = torch.argmax(q_values).view((-1))

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
    ) -> torch.Tensor:
        pass

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        state_batch = batch.state  # (batch_size x state_dim)
        action_batch = batch.action
        # (batch_size x action_dim)
        reward_batch = batch.reward  # (batch_size)
        done_batch = batch.done  # (batch_size)

        batch_size = state_batch.shape[0]
        # sanity check they have same batch_size
        assert reward_batch.shape[0] == batch_size
        assert done_batch.shape[0] == batch_size
        state_action_values = self._Q.get_q_values(
            state_batch=state_batch,
            action_batch=action_batch,
            curr_available_actions_batch=batch.curr_available_actions,
        )  # for duelling, this takes care of the mean subtraction for advantage estimation

        # Compute the Bellman Target
        expected_state_action_values = (
            self._get_next_state_values(batch, batch_size)
            * self._discount_factor
            * (1 - done_batch.float())
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        criterion = torch.nn.MSELoss()
        bellman_loss = criterion(state_action_values, expected_state_action_values)
        if self._is_conservative:
            cql_loss = compute_cql_loss(self._Q, batch, batch_size)
            loss = self._conservative_alpha * cql_loss + bellman_loss
        else:
            loss = bellman_loss

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Target Network Update
        if (self._training_steps + 1) % self._target_update_freq == 0:
            update_target_network(self._Q_target, self._Q, self._soft_update_tau)

        return {
            "loss": torch.abs(state_action_values - expected_state_action_values)
            .mean()
            .item()
        }
