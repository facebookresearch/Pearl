# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import abstractmethod
from typing import Any, cast, Dict, Iterable, List, Optional, Type, Union

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
from pearl.neural_networks.common.utils import (
    init_weights,
    update_target_network,
    update_target_networks,
)

from pearl.neural_networks.common.value_networks import (
    QValueNetwork,
    ValueNetwork,
    VanillaQValueNetwork,
    VanillaValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    DynamicActionActorNetwork,
    VanillaActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from torch import nn, optim


class ActorCriticBase(PolicyLearner):
    """
    A base class for all actor-critic based policy learners.

    Many components that are common to all actor-critic methods have been put in this base class.
    These include:

    - actor and critic network initializations (optionally with corresponding target networks).
    - `act`, `reset` and `learn_batch` methods.
    - Utility functions used by many actor-critic methods.
    """

    def __init__(
        self,
        state_dim: int,
        exploration_module: ExplorationModule,
        actor_hidden_dims: List[int],
        critic_hidden_dims: Optional[List[int]] = None,
        action_space: Optional[ActionSpace] = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: Type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: Union[
            Type[ValueNetwork], Type[QValueNetwork]
        ] = VanillaQValueNetwork,
        use_actor_target: bool = False,
        use_critic_target: bool = False,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        use_twin_critic: bool = False,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        is_action_continuous: bool = False,
        on_policy: bool = False,
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        super(ActorCriticBase, self).__init__(
            on_policy=on_policy,
            is_action_continuous=is_action_continuous,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
            action_representation_module=action_representation_module,
            action_space=action_space,
        )
        self._state_dim = state_dim
        self._use_actor_target = use_actor_target
        self._use_critic_target = use_critic_target
        self._use_twin_critic = use_twin_critic
        self._use_critic: bool = critic_hidden_dims is not None

        self._action_dim: int = (
            self.action_representation_module.representation_dim
            if self.is_action_continuous
            else self.action_representation_module.max_number_actions
        )

        # actor network takes state as input and outputs an action vector
        self._actor: nn.Module = actor_network_type(
            input_dim=state_dim + self._action_dim
            if actor_network_type is DynamicActionActorNetwork
            else state_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=1
            if actor_network_type is DynamicActionActorNetwork
            else self._action_dim,
            action_space=action_space,
        )
        self._actor.apply(init_weights)
        self._actor_optimizer = optim.AdamW(
            [
                {
                    "params": self._actor.parameters(),
                    "lr": actor_learning_rate,
                    "amsgrad": True,
                },
            ]
        )
        self._actor_soft_update_tau = actor_soft_update_tau
        if self._use_actor_target:
            self._actor_target: nn.Module = actor_network_type(
                input_dim=state_dim + self._action_dim
                if actor_network_type is DynamicActionActorNetwork
                else state_dim,
                hidden_dims=actor_hidden_dims,
                output_dim=1
                if actor_network_type is DynamicActionActorNetwork
                else self._action_dim,
                action_space=action_space,
            )
            update_target_network(self._actor_target, self._actor, tau=1)

        self._critic_soft_update_tau = critic_soft_update_tau
        if self._use_critic:
            self._critic: nn.Module = make_critic(
                state_dim=self._state_dim,
                action_dim=self._action_dim,
                hidden_dims=critic_hidden_dims,
                use_twin_critic=use_twin_critic,
                network_type=critic_network_type,
            )
            self._critic_optimizer: optim.Optimizer = optim.AdamW(
                [
                    {
                        "params": self._critic.parameters(),
                        "lr": critic_learning_rate,
                        "amsgrad": True,
                    },
                ]
            )
            if self._use_critic_target:
                self._critic_target: nn.Module = make_critic(
                    state_dim=self._state_dim,
                    action_dim=self._action_dim,
                    hidden_dims=critic_hidden_dims,
                    use_twin_critic=use_twin_critic,
                    network_type=critic_network_type,
                )
                update_critic_target_network(
                    self._critic_target,
                    self._critic,
                    use_twin_critic,
                    1,
                )

        self._discount_factor = discount_factor

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        self._actor_optimizer.add_param_group({"params": value.parameters()})
        if self._use_critic:
            self._critic_optimizer.add_param_group({"params": value.parameters()})
        self._history_summarization_module = value

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        """
        Determines an action based on the policy network and optionally the exploration module.
        This function can operate in two modes: exploit or explore. The mode is determined by the
        `exploit` parameter.

        - If `exploit` is True, the function returns an action determined solely by the policy
        network.
        - If `exploit` is False, the function first calculates an `exploit_action` using the policy
        network. This action is then passed to the exploration module, along with additional
        arguments specific to the exploration module in use. The exploration module then generates
        an action that strikes a balance between exploration and exploitation.

        Args:
            subjective_state (SubjectiveState): Subjective state of the agent.
            available_action_space (ActionSpace): Set of eligible actions.
            exploit (bool, optional): Determines the mode of operation. If True, the function
            operates in exploit mode. If False, it operates in explore mode. Defaults to False.
        Returns:
            Action: An action (decision made by the agent in the given subjective state)
            that balances between exploration and exploitation, depending on the mode
            specified by the user. The returned action is from the available action space.
        """
        # Step 1: compute exploit_action
        # (action computed by actor network; and without any exploration)
        with torch.no_grad():
            if self.is_action_continuous:
                exploit_action = self._actor.sample_action(subjective_state)
                action_probabilities = None
            else:
                assert isinstance(available_action_space, DiscreteActionSpace)
                actions = self.action_representation_module(
                    available_action_space.actions_batch
                )
                action_probabilities = self._actor.get_policy_distribution(
                    state_batch=subjective_state,
                    available_actions=actions,
                )
                # (action_space_size)
                exploit_action = torch.argmax(action_probabilities)

        # Step 2: return exploit action if no exploration,
        # else pass through the exploration module
        if exploit:
            return exploit_action

        # TODO: carefully check if safe action space is integrated with the exploration module
        return self._exploration_module.act(
            exploit_action=exploit_action,
            action_space=available_action_space,
            subjective_state=subjective_state,
            values=action_probabilities,
        )

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        Trains the actor and critic networks using a batch of transitions.
        This method performs the following steps:

        1. Updates the actor network with the input batch of transitions.
        2. Updates the critic network with the input batch of transitions.
        3. If using target network for critics (i.e. `use_critic_target` argument is True), the
        function updates the critic target network.
        4. If using target network for policy (i.e. `use_actor_target` argument is True), the
        function updates the actor target network.

        Note: While this method provides a general approach to actor-critic methods, specific
        algorithms may override it to introduce unique behaviors. For instance, the TD3 algorithm
        updates the actor network less frequently than the critic network.

        Args:
            batch (TransitionBatch): Batch of transitions to use for actor and critic updates.
        Returns:
            Dict[str, Any]: A dictionary containing the loss reports from the critic
            and actor updates. These can be useful to track for debugging purposes.
        """
        actor_loss = self._actor_loss(batch)
        self._actor_optimizer.zero_grad()
        if self._use_critic:
            critic_loss = self._critic_loss(batch)
            self._critic_optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            self._actor_optimizer.step()
            self._critic_optimizer.step()
            report = {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
            }
        else:
            actor_loss.backward()
            self._actor_optimizer.step()
            report = {"actor_loss": actor_loss.item()}

        if self._use_critic_target:
            update_critic_target_network(
                self._critic_target,
                self._critic,
                self._use_twin_critic,
                self._critic_soft_update_tau,
            )
        if self._use_actor_target:
            update_target_network(
                self._actor_target,
                self._actor,
                self._actor_soft_update_tau,
            )
        return report

    def preprocess_batch(self, batch: TransitionBatch) -> TransitionBatch:
        """
        Preprocesses a batch of transitions before learning on it.
        This method should be called in the learner process.
        """
        # change reward to be the lambda_constraint weighted sum of reward and cost
        if hasattr(self.safety_module, "lambda_constraint"):
            batch.reward = (
                batch.reward - self.safety_module.lambda_constraint * batch.cost
            )
        batch = super().preprocess_batch(batch)

        return batch

    @abstractmethod
    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Abstract method for implementing the algorithm-specific logic for updating the actor
        network. This method must be implemented by any concrete subclass to provide the specific
        logic for updating the actor network based on the algorithm implemented by the subclass.
        Args:
            batch (TransitionBatch): A batch of transitions used for updating the actor network.
        Returns:
            loss (Tensor): The actor loss.
        """
        pass

    @abstractmethod
    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Abstract method for implementing the algorithm-specific logic for updating the critic
        network. This method must be implemented by any concrete subclass to provide the specific
        logic for updating the critic network based on the algorithm implemented by the subclass.
        Args:
            batch (TransitionBatch): A batch of transitions used for updating the actor network.
        Returns:
            loss (Tensor): The critic loss.
        """
        pass


def make_critic(
    state_dim: int,
    hidden_dims: Optional[Iterable[int]],
    use_twin_critic: bool,
    network_type: Union[Type[ValueNetwork], Type[QValueNetwork]],
    action_dim: Optional[int] = None,
) -> nn.Module:
    if use_twin_critic:
        assert action_dim is not None
        assert hidden_dims is not None
        assert issubclass(
            network_type, QValueNetwork
        ), "network_type must be a subclass of QValueNetwork when use_twin_critic is True"

        # cast network_type to get around static Pyre type checking; the runtime check with
        # `issubclass` ensures the network type is a sublcass of QValueNetwork
        network_type = cast(Type[QValueNetwork], network_type)

        return TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            network_type=network_type,
            init_fn=init_weights,
        )
    else:
        if network_type == VanillaQValueNetwork:
            # pyre-ignore[45]:
            # Pyre does not know that `network_type` is asserted to be concrete
            return network_type(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )
        elif network_type == VanillaValueNetwork:
            # pyre-ignore[45]:
            # Pyre does not know that `network_type` is asserted to be concrete
            return network_type(
                input_dim=state_dim, hidden_dims=hidden_dims, output_dim=1
            )
        else:
            raise NotImplementedError(
                "Unknown network type. The code needs to be refactored to support this."
            )


def update_critic_target_network(
    target_network: nn.Module, network: nn.Module, use_twin_critic: bool, tau: float
) -> None:
    if use_twin_critic:
        update_target_networks(
            target_network._critic_networks_combined,
            network._critic_networks_combined,
            tau=tau,
        )
    else:
        update_target_network(
            target_network._model,
            network._model,
            tau=tau,
        )


def single_critic_state_value_loss(
    state_batch: torch.Tensor,
    expected_target_batch: torch.Tensor,
    critic: nn.Module,
) -> torch.Tensor:
    """
    Performs a single optimization step on a (value) critic network using the input batch of states.
    This method calculates the mean squared error loss between the predicted state values from the
    critic network and the input target estimates. It then updates the critic network using the
    provided optimizer.
    Args:
        state_batch (torch.Tensor): A batch of states with expected shape
        `(batch_size, state_dim)`.
        expected_target_batch (torch.Tensor): The batch of target estimates
        (i.e., RHS of the Bellman equation) with shape `(batch_size)`.
        critic (nn.Module): The critic network to update.
    Returns:
        loss (torch.Tensor): The mean squared error loss for state-value prediction
    """
    if not isinstance(critic, ValueNetwork):
        raise TypeError(
            "critic in the `single_critic_state_value_update` method must be an instance of "
            "ValueNetwork"
        )
    vs = critic(state_batch)
    criterion = torch.nn.MSELoss()
    loss = criterion(
        vs.reshape_as(expected_target_batch), expected_target_batch.detach()
    )
    return loss


def twin_critic_action_value_loss(
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    expected_target_batch: torch.Tensor,
    critic: TwinCritic,
) -> torch.Tensor:
    """
    Performs a single optimization step on the twin critic networks using the input
    batch of states and actions.
    This method calculates the mean squared error loss between the predicted Q-values from both
    critic networks and the input target estimates. It then updates the critic networks using the
    provided optimizer.

    Args:
        state_batch (torch.Tensor): A batch of states with expected shape
        `(batch_size, state_dim)`.
        action_batch (torch.Tensor): A batch of actions with expected shape
        `(batch_size, action_dim)`.
        expected_target_batch (torch.Tensor): The batch of target estimates
        (i.e. RHS of the Bellman equation) with shape `(batch_size)`.
        critic (TwinCritic): The twin critic network to update.
    Returns:
        loss (torch.Tensor): The mean squared error loss for action-value prediction
    """

    criterion = torch.nn.MSELoss()
    q_1, q_2 = critic.get_q_values(state_batch, action_batch)
    loss = criterion(
        q_1.reshape_as(expected_target_batch), expected_target_batch.detach()
    ) + criterion(q_2.reshape_as(expected_target_batch), expected_target_batch.detach())
    return loss
