from typing import Any, Dict, Iterable, Type

import torch
from pearl.api.action_space import ActionSpace

from pearl.neural_networks.common.utils import update_target_network

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
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    OffPolicyActorCritic,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import nn


class DeepDeterministicPolicyGradient(OffPolicyActorCritic):
    """
    A Class for Deep Deterministic Deep Policy Gradient policy learner.
    paper: https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        exploration_module: ExplorationModule,
        hidden_dims: Iterable[int],
        critic_learning_rate: float = 1e-3,
        actor_learning_rate: float = 1e-3,
        batch_size: int = 256,
        actor_network_type: ActorNetworkType = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        training_rounds: int = 1,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
    ) -> None:
        super(DeepDeterministicPolicyGradient, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            critic_learning_rate=critic_learning_rate,
            actor_learning_rate=actor_learning_rate,
            exploration_module=exploration_module,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            critic_soft_update_tau=critic_soft_update_tau,
            is_action_continuous=True,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
        )

        # target of actor network: DDPG and TD3 use an additional target network for actor networks
        # pyre-ignore
        self._actor_target: nn.Module = actor_network_type(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=self._action_dim,  # OffPolicyActorCritic base class has action_dim as an attribute
            action_space=action_space,
        )

        # initialize weights of actor target network to be the same as the actor network
        # actor network is instantiated by the OffPolicyActorCritic base class
        self._actor_target.load_state_dict(self._actor.state_dict())

        self._actor_soft_update_tau = actor_soft_update_tau
        self._rounds = 0

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        # actor and critic updates, and updates of targets of twin critics are performed
        # using the 'learn_batch' method in the OffPolicyActorCritic base class
        super(DeepDeterministicPolicyGradient, self).learn_batch(batch)

        # soft update target of actor network
        update_target_network(
            self._actor_target, self._actor, self._actor_soft_update_tau
        )
        return {}

    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        # sample a batch of actions from the actor network; shape (batch_size, action_dim)
        action_batch = self._actor.sample_action(batch.state)

        # samples q values for (batch.state, action_batch) from twin critics
        q1, q2 = self._twin_critics.get_twin_critic_values(
            state_batch=batch.state, action_batch=action_batch
        )

        # clipped double q learning (reduce overestimation bias); shape (batch_size)
        q = torch.minimum(q1, q2)

        # optimization objective: optimize actor to maximize Q(s, a)
        loss = -q.mean()

        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return {"actor_loss": loss.mean().item()}

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        with torch.no_grad():
            # sample a batch of next actions from target actor network; shape (batch_size, action_dim)
            next_action = self._actor_target.sample_action(batch.next_state)

            # get q values of (batch.next_state, next_action) from targets of twin critic
            next_q1, next_q2 = self._targets_of_twin_critics.get_twin_critic_values(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Optional[Tensor]`.
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias); shape (batch_size)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network), Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.done.float())
            ) + batch.reward  # shape (batch_size)

        # update twin critics towards bellman target
        loss_critic_update = self.twin_critic_update(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target=expected_state_action_values,
        )

        return loss_critic_update
