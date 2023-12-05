# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.neural_networks.common.value_networks import QValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    DynamicActionActorNetwork,
)

from pearl.utils.instantiations.spaces.box_action import BoxActionSpace

from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

import torch

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
    VanillaQValueNetwork,
    VanillaValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    VanillaActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from torch import nn, optim


class ActorCriticBase(PolicyLearner):
    """
    A base class for all actor-critic based policy learners.
    Many components are common to actor-critic methods.
        - Actor and critic (as well as target networks) network initializations.
        - Act, reset and learn_batch methods.
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
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
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

        return self._exploration_module.act(
            exploit_action=exploit_action,
            action_space=available_action_space,
            subjective_state=subjective_state,
            values=action_probabilities,
        )

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        self._critic_learn_batch(batch)  # update critic
        self._actor_learn_batch(batch)  # update actor

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
        return {}

    @abstractmethod
    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        pass


def make_critic(
    state_dim: int,
    hidden_dims: Optional[Iterable[int]],
    use_twin_critic: bool,
    network_type: Type[QValueNetwork],
    action_dim: Optional[int] = None,
) -> nn.Module:
    if use_twin_critic:
        assert action_dim is not None
        assert hidden_dims is not None
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


def single_critic_state_value_update(
    state_batch: torch.Tensor,
    expected_target_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    critic: nn.Module,
) -> Dict[str, Any]:
    vs = critic(state_batch)
    # critic loss
    criterion = torch.nn.MSELoss()
    loss = criterion(
        vs.reshape_as(expected_target_batch), expected_target_batch.detach()
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"critic_loss": loss.mean().item()}


def twin_critic_action_value_update(
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    expected_target_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    critic: TwinCritic,
) -> Dict[str, torch.Tensor]:
    """
    Performs an optimization step on the twin critic networks.

    Args:
        state_batch: a batch of states with shape (batch_size, state_dim)
        action_batch: a batch of actions with shape (batch_size, action_dim)
        expected_target: the batch of target estimates for Bellman equation.
        optimizer: the optimizer to use for the update.
        critic: the critic network to update.
    Returns:
        Dict[str, torch.Tensor]: mean loss and individual critic losses.
    """

    criterion = torch.nn.MSELoss()
    optimizer.zero_grad()
    q_1, q_2 = critic.get_q_values(state_batch, action_batch)
    loss = criterion(
        q_1.reshape_as(expected_target_batch), expected_target_batch.detach()
    ) + criterion(q_2.reshape_as(expected_target_batch), expected_target_batch.detach())
    loss.backward()
    optimizer.step()

    return {
        "critic_mean_loss": loss.item(),
        "critic_1_values": q_1.mean().item(),
        "critic_2_values": q_2.mean().item(),
    }
