# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import copy
from abc import abstractmethod
from typing import Any, List, Optional

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
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    DuelingQValueNetwork,
    QValueNetwork,
    TwoTowerQValueNetwork,
    VanillaQValueMultiHeadNetwork,
    VanillaQValueNetwork,
)

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch

from pearl.utils.functional_utils.learning.loss_fn_utils import compute_cql_loss
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.utils.module_utils import modules_have_similar_state_dict
from torch import optim


# TODO: Only support discrete action space problems for now and assumes Gym action space.
# Currently available actions is not used. Needs to be updated once we know the input structure
# of production stack on this param.
class DeepTDLearning(PolicyLearner):
    """
    An Abstract Class for Deep Temporal Difference learning.
    """

    def __init__(
        self,
        exploration_module: ExplorationModule,
        on_policy: bool,
        state_dim: int | None = None,
        action_space: ActionSpace | None = None,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        target_update_freq: int = 10,
        soft_update_tau: float = 0.1,
        is_conservative: bool = False,
        conservative_alpha: float | None = 2.0,
        network_type: type[QValueNetwork] = VanillaQValueNetwork,
        state_output_dim: int | None = None,
        action_output_dim: int | None = None,
        state_hidden_dims: list[int] | None = None,
        action_hidden_dims: list[int] | None = None,
        network_instance: QValueNetwork | None = None,
        action_representation_module: ActionRepresentationModule | None = None,
        optimizer: Optional[optim.Optimizer] = None,
        **kwargs: Any,
    ) -> None:
        """Constructs a DeepTDLearning based policy learner. DeepTDLearning is the base class
        for all value based (i.e. temporal difference learning based) algorithms.

        Args:
            state_dim: Dimension of the state space.
            exploration_module (ExplorationModule, optional): Optional exploration module used by
                the `act` function to trade-off between exploitation and exploration.
                Defaults to None.
            action_space (ActionSpace, optional): Action space of the problem. It is kept optional
                to allow for the use of dynamic action spaces (see `learn_batch` and `act`
                functions). Defaults to None.
            hidden_dims (List[int], optional): Hidden dimensions of the default `QValueNetwork`
                (taken to be `VanillaQValueNetwork`). Defaults to None.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.001.
                Note: We use AdamW as default for all value based methods.
            discount_factor (float): Discount factor for TD updates. Defaults to 0.99.
            training_rounds (int): Number of gradient updates per environment step.
                Defaults to 100.
            batch_size (int): Sample size for mini-batch gradient updates. Defaults to 128.
            target_update_freq (int): Frequency at which the target network is updated.
                Defaults to 100.
            soft_update_tau (float): Coefficient for soft updates to the target networks.
                Defaults to 0.01.
            is_conservative (bool): Whether to use conservative updates for offline learning.
                Defaults to False.
            conservative_alpha (float, optional): Alpha parameter for conservative updates.
                Defaults to 2.0.
            network_type (Type[QValueNetwork]): Network type for the Q-value network. Defaults to
                `VanillaQValueNetwork`. This means that by default, an instance of the class
                `VanillaQValueNetwork` (or the specified `network_type` class) is created and used
                for learning.
            network_instance (QValueNetwork, optional): A network instance to be used as the
                Q-value network. Defaults to None.
                Note: This is an alternative to specifying a `network_type`. If provided, the
                specified `network_type` is ignored and the input `network_instance` is used for
                learning. Allows for custom implementations of Q-value networks.
            action_representation_module (ActionRepresentationModule, optional): Optional module to
                represent actions as a feature vector. Typically specified at the agent level.
                Defaults to None.
        """
        super().__init__(
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
            assert state_dim is not None
            assert hidden_dims is not None
            if network_type is TwoTowerQValueNetwork:
                return network_type(
                    state_dim=state_dim,
                    # pyre-fixme[6]: For 2nd argument expected `int` but got
                    #  `Union[Tensor, Module]`.
                    action_dim=self.action_representation_module.representation_dim,
                    hidden_dims=hidden_dims,
                    state_output_dim=state_output_dim,
                    action_output_dim=action_output_dim,
                    state_hidden_dims=state_hidden_dims,
                    action_hidden_dims=action_hidden_dims,
                    output_dim=1,
                )
            elif network_type is VanillaQValueMultiHeadNetwork:
                return network_type(
                    state_dim=state_dim,
                    # pyre-fixme[6]: For 2nd argument expected `int` but got
                    #  `Union[Tensor, Module]`.
                    action_dim=self.action_representation_module.representation_dim,
                    hidden_dims=hidden_dims,
                    # pyre-fixme[6]: For 3rd argument expected `int` but got
                    # `Union[Tensor, Module]`.
                    output_dim=self.action_representation_module.max_number_actions,
                )
            else:
                assert (
                    network_type is VanillaQValueNetwork
                    or network_type is DuelingQValueNetwork
                )
                return network_type(
                    state_dim=state_dim,
                    # pyre-fixme[6]: For 2nd argument expected `int` but got
                    #  `Union[Tensor, Module]`.
                    action_dim=self.action_representation_module.representation_dim,
                    hidden_dims=hidden_dims,
                    output_dim=1,
                )

        if network_instance is not None:
            self._Q: QValueNetwork = network_instance
        else:
            self._Q = make_specified_network()

        self._Q_target: QValueNetwork = copy.deepcopy(self._Q)
        if optimizer is not None:
            self._optimizer = optimizer
        else:
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
        """
        Selects an action from the available action space balancing between exploration and
        exploitation.
        This action can be (i) an 'exploit action', i.e. the optimal action given estimate of the
        Q values or (ii) an 'exploratory action' obtained using the specified `exploration_module`.

        Args:
            subjective_state (SubjectiveState): Current subjective state.
            available_action_space (ActionSpace): Available action space at the current state.
                Note that Pearl allows for action spaces to change dynamically.
            exploit (bool): When set to True, we output the exploit action (no exploration).
                When set to False, the specified `exploration_module` is used to balance
                between exploration and exploitation. Defaults to False.

        Returns:
            Action: An action from the available action space.
        """
        # TODO: Assumes gym action space.
        # Fix the available action space.
        assert isinstance(available_action_space, DiscreteActionSpace)
        with torch.no_grad():
            batched_actions_representation = self.action_representation_module(
                available_action_space.actions_batch.to(subjective_state)
            ).unsqueeze(0)  # (1 x number of actions x action_dim)

            q_values = self._Q.get_q_values(
                subjective_state.unsqueeze(0),  # (1 x state_dim)
                batched_actions_representation,
            )  # (1 x number of actions)
            # this does a forward pass since all avaialble
            # actions are already stacked together
            q_values = q_values.squeeze(0)  # (number of actions)
            exploit_action_index = torch.argmax(q_values)
            exploit_action = available_action_space.actions[exploit_action_index]

        if exploit:
            return exploit_action

        assert self.exploration_module is not None
        return self.exploration_module.act(
            subjective_state=subjective_state,
            action_space=available_action_space,
            exploit_action=exploit_action,
            values=q_values,
        )

    @abstractmethod
    def get_next_state_values(
        self, batch: TransitionBatch, batch_size: int
    ) -> torch.Tensor:
        """
        For a given batch of transitions, returns Q-value targets for the Bellman equation.
        Child classes should implement this method.

        For example, this method in DQN returns
        "max_{action in available_action_space} Q(next_state, action)".
        """
        pass

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        """
        Batch learning with TD(0) style updates. Different implementations of the
        `get_next_state_values` function correspond to the different RL algorithm implementations,
        for example TD learning, DQN, Double DQN, Duelling DQN etc.

        Args:
            batch (TransitionBatch): batch of transitions
        Returns:
            Dict[str, Any]: dictionary with loss as the mean bellman error (across the batch).
        """
        state_batch = batch.state  # (batch_size x state_dim)
        action_batch = batch.action  # (batch_size x action_dim)
        reward_batch = batch.reward  # (batch_size)
        terminated_batch = batch.terminated  # (batch_size)

        batch_size = state_batch.shape[0]
        # sanity check they have same batch_size
        assert reward_batch.shape[0] == batch_size
        assert terminated_batch.shape[0] == batch_size

        state_action_values = self._Q.get_q_values(
            state_batch=state_batch,
            action_batch=action_batch,
            curr_available_actions_batch=batch.curr_available_actions,
        )  # (batch_size)
        # for duelling dqn, specifying the `curr_available_actions_batch` field takes care of
        # the mean subtraction for advantage estimation

        # Compute the Bellman Target
        expected_state_action_values = (
            self.get_next_state_values(batch, batch_size)
            * self._discount_factor
            * (1 - terminated_batch.float())
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        criterion = torch.nn.MSELoss()
        bellman_loss = criterion(state_action_values, expected_state_action_values)

        # Conservative TD updates for offline learning.
        if self._is_conservative:
            cql_loss = compute_cql_loss(self._Q, batch, batch_size)
            # pyre-fixme[58]: `*` is not supported for operand types
            #  `Optional[float]` and `Tensor`.
            loss = self._conservative_alpha * cql_loss + bellman_loss
        else:
            loss = bellman_loss

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Target network update
        if (self._training_steps + 1) % self._target_update_freq == 0:
            update_target_network(self._Q_target, self._Q, self._soft_update_tau)

        return {
            "loss": torch.abs(state_action_values - expected_state_action_values)
            .mean()
            .item()
        }

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two DeepTDLearning instances for equality,
        checking attributes, Q-networks, and exploration module.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """
        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, DeepTDLearning):
            differences.append("other is not an instance of DeepTDLearning")
        else:
            # Compare attributes
            if self._learning_rate != other._learning_rate:
                differences.append(
                    f"_learning_rate is different: {self._learning_rate} vs {other._learning_rate}"
                )
            if self._discount_factor != other._discount_factor:
                differences.append(
                    f"_discount_factor is different: {self._discount_factor} "
                    + f"vs {other._discount_factor}"
                )
            if self._target_update_freq != other._target_update_freq:
                differences.append(
                    f"_target_update_freq is different: {self._target_update_freq} "
                    + f"vs {other._target_update_freq}"
                )
            if self._soft_update_tau != other._soft_update_tau:
                differences.append(
                    f"_soft_update_tau is different: {self._soft_update_tau} "
                    + f"vs {other._soft_update_tau}"
                )
            if self._is_conservative != other._is_conservative:
                differences.append(
                    f"_is_conservative is different: {self._is_conservative} "
                    + f"vs {other._is_conservative}"
                )
            if self._conservative_alpha != other._conservative_alpha:
                differences.append(
                    f"_conservative_alpha is different: {self._conservative_alpha} "
                    + f"vs {other._conservative_alpha}"
                )

            # Compare Q-networks and target Q-networks using modules_have_similar_state_dict
            if (reason := modules_have_similar_state_dict(self._Q, other._Q)) != "":
                differences.append(f"_Q is different: {reason}")
            if (
                reason := modules_have_similar_state_dict(
                    self._Q_target, other._Q_target
                )
            ) != "":
                differences.append(f"_Q_target is different: {reason}")

        return "\n".join(differences)
