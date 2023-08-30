from typing import Any, Dict, Iterable, Type

import torch
from pearl.core.common.neural_networks.q_value_network import QValueNetwork
from pearl.core.common.neural_networks.utils import (
    update_target_network,
    update_target_networks,
)

from pearl.core.common.neural_networks.value_networks import VanillaQValueNetwork
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.core.sequential_decision_making.neural_networks.actor_networks import (
    ActorNetworkType,
    VanillaContinuousActorNetwork,
)
from pearl.core.sequential_decision_making.policy_learners.ddpg import (
    DeepDeterministicPolicyGradient,
)


class TD3(DeepDeterministicPolicyGradient):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        exploration_module: ExplorationModule = None,
        critic_learning_rate: float = 1e-2,
        actor_learning_rate: float = 1e-3,
        batch_size: int = 500,
        actor_network_type: ActorNetworkType = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        training_rounds: int = 5,
        actor_soft_update_tau: float = 0.05,
        critic_soft_update_tau: float = 0.05,
        discount_factor: float = 0.98,
        actor_update_freq: int = 2,
        learning_action_noise_std: float = 0.1,
        learning_action_noise_clip: float = 0.5,
    ) -> None:
        super(TD3, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            exploration_module=exploration_module,
            critic_learning_rate=critic_learning_rate,
            actor_learning_rate=actor_learning_rate,
            batch_size=batch_size,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            training_rounds=training_rounds,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            num_critic_network=2,
        )
        self._actor_update_freq = actor_update_freq
        self._learning_action_noise_std = learning_action_noise_std
        self._learning_action_noise_clip = learning_action_noise_clip
        self._critic_update_count = 0

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        self._critic_update_count += 1
        report = self._critic_learn_batch(batch)  # critic update

        # see ddpg base class for actor update details
        if self._critic_update_count == self._actor_update_freq:
            report.update(self._actor_learn_batch(batch))  # actor update
            self._critic_update_count = 0  # reset counter

        # update targets of twin critics using soft updates
        update_target_networks(
            self._targets_of_twin_critics._critic_networks_combined,
            self._twin_critics._critic_networks_combined,
            self._critic_soft_update_tau,
        )
        # update target of actor network using soft updates
        update_target_network(
            self._actor_target, self._actor, self._actor_soft_update_tau
        )
        return report

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        with torch.no_grad():

            # sample next_action from actor's target network
            next_action = self._actor_target(batch.next_state)

            # sample clipped gaussian noise
            noise = torch.normal(
                mean=0,
                std=self._learning_action_noise_std,
                size=next_action.size(),
                device=self.device,
            )
            noise = torch.clamp(
                noise,
                -self._learning_action_noise_clip,
                self._learning_action_noise_clip,
            )
            next_action = next_action + noise  # add noise to next_action

            # sample q values of (next_state, next_action) from targets of twin critics
            next_q1, next_q2 = self._targets_of_twin_critics.get_twin_critic_values(
                state_batch=batch.next_state, action_batch=next_action
            )
            next_q = torch.minimum(
                next_q1, next_q2
            )  # clipped double q learning (reduce overestimation bias)

            # compute bellman target
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.done)
            ) + batch.reward  # (batch_size), r + gamma * (min{Q_1(s', a from actor network), Q_2(s', a from actor network)})

        # update twin critics towards bellman target
        loss_critic_update = self._twin_critics.optimize_twin_critics_towards_target(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target=expected_state_action_values,
        )
        return loss_critic_update
