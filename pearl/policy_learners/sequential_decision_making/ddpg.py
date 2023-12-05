from typing import Any, Dict, Iterable, Optional, Type

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace

from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaContinuousActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.common.normal_distribution_exploration import (  # noqa E501
    NormalDistributionExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
    twin_critic_action_value_update,
)
from pearl.replay_buffers.transition import TransitionBatch


class DeepDeterministicPolicyGradient(ActorCriticBase):
    """
    A Class for Deep Deterministic Deep Policy Gradient policy learner.
    paper: https://arxiv.org/pdf/1509.02971.pdf
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
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        super(DeepDeterministicPolicyGradient, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=True,
            use_critic_target=True,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            use_twin_critic=True,  # we need to make this optional to users
            exploration_module=exploration_module
            if exploration_module is not None
            else NormalDistributionExploration(mean=0.0, std_dev=0.1),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=True,
            on_policy=False,
            action_representation_module=action_representation_module,
        )

    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        # sample a batch of actions from the actor network; shape (batch_size, action_dim)
        action_batch = self._actor.sample_action(batch.state)

        # samples q values for (batch.state, action_batch) from twin critics
        q1, q2 = self._critic.get_q_values(
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
            # sample a batch of next actions from target actor network;
            next_action = self._actor_target.sample_action(batch.next_state)
            # (batch_size, action_dim)

            # get q values of (batch.next_state, next_action) from targets of twin critic
            next_q1, next_q2 = self._critic_target.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias); shape (batch_size)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network),
            #                  Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.done.float())
            ) + batch.reward  # shape (batch_size)

        assert isinstance(self._critic, TwinCritic), "DDPG requires TwinCritic critic"

        # update twin critics towards bellman target
        loss_critic_update = twin_critic_action_value_update(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            optimizer=self._critic_optimizer,
            critic=self._critic,
        )

        return loss_critic_update
