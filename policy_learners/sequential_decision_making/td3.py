from typing import Any, Dict, Iterable, Type

import torch
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.utils import (
    update_target_network,
    update_target_networks,
)

from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetworkType,
    VanillaContinuousActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.ddpg import (
    DeepDeterministicPolicyGradient,
)
from pearl.replay_buffers.transition import TransitionBatch


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
        hidden_dims: Iterable[int],
        exploration_module: ExplorationModule,
        critic_learning_rate: float = 1e-3,
        actor_learning_rate: float = 1e-3,
        batch_size: int = 256,
        actor_network_type: ActorNetworkType = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        training_rounds: int = 1,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        actor_update_freq: int = 2,
        actor_update_noise: float = 0.2,
        actor_update_noise_clip: float = 0.5,
    ) -> None:
        super(TD3, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
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
        )
        self._action_space = action_space
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
            update_target_networks(
                self._targets_of_twin_critics._critic_networks_combined,
                self._twin_critics._critic_networks_combined,
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
            low, high = torch.tensor(
                self._action_space.low, device=batch.device  # pyre-ignore
            ), torch.tensor(
                self._action_space.high, device=batch.device  # pyre-ignore
            )

            next_action = torch.clamp(
                next_action + noise, low, high
            )  # shape (batch_size, action_dim)

            # sample q values of (next_state, next_action) from targets of twin critics
            next_q1, next_q2 = self._targets_of_twin_critics.get_twin_critic_values(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Optional[Tensor]`.
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network), Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.done.float())
            ) + batch.reward  # (batch_size)

        # update twin critics towards bellman target
        loss_critic_update = self.twin_critic_update(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target=expected_state_action_values,
        )
        return loss_critic_update
