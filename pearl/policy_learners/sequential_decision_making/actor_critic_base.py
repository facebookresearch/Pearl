# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional

from pearl.utils.instantiations.action_spaces.box import BoxActionSpace

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

import torch

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
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
    ActorNetworkType,
    VanillaActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.action_spaces.discrete import DiscreteActionSpace
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
        action_space: ActionSpace,
        exploration_module: ExplorationModule,
        actor_hidden_dims: Iterable[int],
        critic_hidden_dims: Optional[Iterable[int]],
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: ActorNetworkType = VanillaActorNetwork,
        # pyre-fixme
        critic_network_type=VanillaValueNetwork,
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
    ) -> None:
        super(ActorCriticBase, self).__init__(
            on_policy=on_policy,
            is_action_continuous=is_action_continuous,
            training_rounds=training_rounds,
            batch_size=batch_size,
            exploration_module=exploration_module,
        )
        self._state_dim = state_dim
        self._use_actor_target = use_actor_target
        self._use_critic_target = use_critic_target
        self._use_twin_critic = use_twin_critic
        self._use_critic: bool = critic_hidden_dims is not None

        if isinstance(
            action_space, (gym.spaces.discrete.Discrete, DiscreteActionSpace)
        ):
            # TODO: This assumes OneHotActionRepresentation
            self._action_dim: int = action_space.n
        elif isinstance(action_space, (gym.spaces.box.Box, BoxActionSpace)):
            self._action_dim = action_space.action_dim
        else:
            raise NotImplementedError("Action space not implemented")

        # actor network takes state as input and outputs an action vector
        self._actor: nn.Module = actor_network_type(
            input_dim=state_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=self._action_dim,
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
                input_dim=state_dim,
                hidden_dims=actor_hidden_dims,
                output_dim=self._action_dim,
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
            self._critic_optimizer = optim.AdamW(
                [
                    {
                        "params": self._critic.parameters(),
                        "lr": critic_learning_rate,
                        "amsgrad": True,
                    },
                ]
            )
            if self._use_critic_target:
                self._critic_target = make_critic(
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

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        # TODO: Assumes subjective state is a torch tensor and gym action space.
        # Fix the available action space.

        # Step 1: compute exploit_action
        # (action computed by actor network; and without any exploration)
        with torch.no_grad():
            if self.is_action_continuous:
                exploit_action = self._actor.sample_action(subjective_state)
                action_probabilities = None
            else:
                action_probabilities = self._actor(subjective_state)
                # (action_space_size, 1)
                exploit_action = torch.argmax(action_probabilities).view((-1)).item()

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

    def _get_action_prob(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        # pyre-fixme[2]: Parameter must be annotated.
        actor=None,
    ) -> torch.Tensor:
        if actor is None:
            action_probs = self._actor(state_batch)
        else:
            action_probs = actor(state_batch)
        # TODO action_batch is one-hot encoding vectors
        return torch.sum(action_probs * action_batch, dim=1, keepdim=True)

    @abstractmethod
    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        pass


# pyre-fixme
def make_critic(
    state_dim: int,
    hidden_dims: Optional[Iterable[int]],
    use_twin_critic: bool,
    # pyre-fixme
    network_type,
    action_dim: Optional[int] = None,
):
    if use_twin_critic:
        return TwinCritic(
            state_dim=state_dim,
            # pyre-fixme
            action_dim=action_dim,
            # pyre-fixme
            hidden_dims=hidden_dims,
            network_type=network_type,
            init_fn=init_weights,
        )
    else:
        if network_type == VanillaQValueNetwork:
            return network_type(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )
        elif network_type == VanillaValueNetwork:
            return network_type(
                input_dim=state_dim, hidden_dims=hidden_dims, output_dim=1
            )
        else:
            raise NotImplementedError(
                "Unknown network type. The code needs to be refactored to support this."
            )


# pyre-fixme
def update_critic_target_network(target_network, network, use_twin_critic, tau):
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
    loss = criterion(vs, expected_target_batch.detach())
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
    loss = criterion(q_1, expected_target_batch.detach()) + criterion(
        q_2, expected_target_batch.detach()
    )
    loss.backward()
    optimizer.step()

    return {
        "critic_mean_loss": loss.item(),
        "critic_1_values": q_1.mean().item(),
        "critic_2_values": q_2.mean().item(),
    }
