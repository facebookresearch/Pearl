from typing import Any, Dict, Iterable, Optional, Type

import torch
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.utils import update_target_network

from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaContinuousActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    twin_critic_action_value_update,
    update_critic_target_network,
)
from pearl.policy_learners.sequential_decision_making.ddpg import (
    DeepDeterministicPolicyGradient,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace


class TD3(DeepDeterministicPolicyGradient):
    """
    TD3 uses a deterministic actor, Twin critics, and a delayed actor update.
        - An exploration module is used with deterministic actors.
        - To avoid exploration, use NoExploration module.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: Iterable[int],
        critic_hidden_dims: Iterable[int],
        exploration_module: Optional[ExplorationModule] = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: Type[ActorNetwork] = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        actor_update_noise: float = 0.2,
        actor_update_noise_clip: float = 0.5,
    ) -> None:
        assert isinstance(action_space, BoxActionSpace)
        super(TD3, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            exploration_module=exploration_module,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
        )
        self._action_space: BoxActionSpace = action_space
        self._actor_update_freq = actor_update_freq
        self._actor_update_noise = actor_update_noise
        self._actor_update_noise_clip = actor_update_noise_clip
        self._critic_update_count = 0

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        self._critic_learn_batch(batch)  # critic update
        self._critic_update_count += 1

        # delayed actor update
        if self._critic_update_count % self._actor_update_freq == 0:
            # see ddpg base class for actor update details
            self._actor_learn_batch(batch)

            # update targets of twin critics using soft updates
            update_critic_target_network(
                self._critic_target,
                self._critic,
                self._use_twin_critic,
                self._critic_soft_update_tau,
            )
            # update target of actor network using soft updates
            update_target_network(
                self._actor_target, self._actor, self._actor_soft_update_tau
            )

        return {}

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        with torch.no_grad():
            # sample next_action from actor's target network; shape (batch_size, action_dim)
            next_action = self._actor_target.sample_action(batch.next_state)

            # sample clipped gaussian noise
            noise = torch.normal(
                mean=0,
                std=self._actor_update_noise,
                size=next_action.size(),
                device=batch.device,
            )

            noise = torch.clamp(
                noise,
                -self._actor_update_noise_clip,
                self._actor_update_noise_clip,
            )  # shape (batch_size, action_dim)

            # add clipped noise to next_action
            low = torch.tensor(self._action_space.low, device=batch.device)
            high = torch.tensor(self._action_space.high, device=batch.device)

            next_action = torch.clamp(
                next_action + noise, low, high
            )  # shape (batch_size, action_dim)

            # sample q values of (next_state, next_action) from targets of twin critics
            next_q1, next_q2 = self._critic_target.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network),
            #                  Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.done.float())
            ) + batch.reward  # (batch_size)

        # update twin critics towards bellman target
        loss_critic_update = twin_critic_action_value_update(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            optimizer=self._critic_optimizer,
            # pyre-fixme
            critic=self._critic,
        )
        return loss_critic_update


class RCTD3(TD3):
    """
    Placeholder for future implementation of RCTD3
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: Iterable[int],
        critic_hidden_dims: Iterable[int],
        exploration_module: Optional[ExplorationModule] = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: Type[ActorNetwork] = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        actor_update_noise: float = 0.2,
        actor_update_noise_clip: float = 0.5,
    ) -> None:
        super(RCTD3, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            exploration_module=exploration_module,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            actor_update_freq=actor_update_freq,
            actor_update_noise=actor_update_noise,
            actor_update_noise_clip=actor_update_noise_clip,
        )
