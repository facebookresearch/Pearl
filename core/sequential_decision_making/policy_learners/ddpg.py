from typing import Any, Dict, Iterable, Type

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.core.common.policy_learners.policy_learner import PolicyLearner

from pearl.neural_networks.common.utils import (
    init_weights,
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
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import optim


class DeepDeterministicPolicyGradient(PolicyLearner):
    """
    A Class for Deep Deterministic Deep Policy Gradient policy learner.
    paper: https://arxiv.org/pdf/1509.02971.pdf

    num_critic_network default to 2, because performance of one critic is 10X slower than twin critic
    Users are free to play with different number here on performance difference
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        exploration_module: ExplorationModule,
        hidden_dims: Iterable[int],
        critic_learning_rate: float = 1e-2,
        actor_learning_rate: float = 1e-3,
        batch_size: int = 500,
        actor_network_type: ActorNetworkType = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        training_rounds: int = 5,
        actor_soft_update_tau: float = 0.05,
        critic_soft_update_tau: float = 0.05,
        discount_factor: float = 0.98,
        num_critic_network: int = 2,
    ) -> None:
        super(DeepDeterministicPolicyGradient, self).__init__(
            exploration_module=exploration_module,
            training_rounds=training_rounds,
            batch_size=batch_size,
            on_policy=False,
            is_action_continuous=True,
        )
        self._state_dim = state_dim
        self._action_dim = action_dim

        def make_specified_actor_network():
            return actor_network_type(
                input_dim=state_dim,
                hidden_dims=hidden_dims,
                output_dim=action_dim,
            )

        # actor network takes state as input and outputs an action vector
        self._actor = make_specified_actor_network()
        self._actor_target = make_specified_actor_network()  # target of actor network
        self._actor.apply(init_weights)
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = optim.AdamW(
            self._actor.parameters(), lr=actor_learning_rate, amsgrad=True
        )

        # twin critic: using two separate critic networks to reduce overestimation bias
        # optimizers of two critics are alredy initialized in TwinCritic
        self._twin_critics = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=critic_learning_rate,
            network_type=critic_network_type,
            init_fn=init_weights,
        )

        # target networks of twin critics
        self._targets_of_twin_critics = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=critic_learning_rate,
            network_type=critic_network_type,
            init_fn=init_weights,
        )

        # target networks are initialized to parameters of the source network (tau is set to 1)
        update_target_networks(
            self._targets_of_twin_critics._critic_networks_combined,
            self._twin_critics._critic_networks_combined,
            tau=1,
        )

        self._actor_soft_update_tau = actor_soft_update_tau
        self._critic_soft_update_tau = critic_soft_update_tau
        self._discount_factor = discount_factor

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace = None,
        exploit: bool = False,
    ) -> Action:
        with torch.no_grad():
            subjective_state_tensor = (
                torch.tensor(subjective_state)
                .view((-1, self._state_dim))
                .to(self.device)
            )  # (batch_size x state_dim)
            exploit_action = self._actor(
                subjective_state_tensor
            )  # (batch_size x action_dim)

        if exploit:
            return exploit_action

        return self._exploration_module.act(
            exploit_action=exploit_action,
        )

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        report = self._critic_learn_batch(batch)  # critic update
        report.update(self._actor_learn_batch(batch))  # actor update

        # update targets of twin critics using soft updates
        update_target_networks(
            self._targets_of_twin_critics._critic_networks_combined,
            self._twin_critics._critic_networks_combined,
            self._critic_soft_update_tau,
        )
        # update target of actor network using soft update
        update_target_network(
            self._actor_target, self._actor, self._actor_soft_update_tau
        )
        return report

    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        action_batch = self._actor(batch.state)
        # samples q values for (batch.state, action_batch) from twin critics
        q1, q2 = self._twin_critics.get_twin_critic_values(
            state_batch=batch.state, action_batch=action_batch
        )
        # optimization objective: optimize pi(.|s) to maximize Q(s, a)
        loss = -torch.minimum(
            q1, q2
        ).mean()  # clipped double q learning (reduce overestimation bias)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return {"actor_loss": loss.mean().item()}

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        with torch.no_grad():

            next_action = self._actor_target(
                batch.next_state
            )  # sample next action from target actor network

            # get q values of (batch.next_state, next_action) from targets of twin critic
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
