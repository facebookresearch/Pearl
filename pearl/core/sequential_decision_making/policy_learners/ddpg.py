from typing import Any, Dict, Iterable

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.core.common.neural_networks.nplets_critic import NpletsCritic

from pearl.core.common.neural_networks.utils import init_weights, update_target_network

from pearl.core.common.neural_networks.value_networks import (
    StateActionValueNetworkType,
    VanillaStateActionValueNetwork,
)
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.core.common.policy_learners.policy_learner import PolicyLearner
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.core.sequential_decision_making.neural_networks.actor_networks import (
    ActorNetworkType,
    VanillaContinuousActorNetwork,
)
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
        critic_network_type: StateActionValueNetworkType = VanillaStateActionValueNetwork,
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

        # actor network takes state and output action vector
        self._actor = make_specified_actor_network()
        self._actor_target = make_specified_actor_network()
        self._actor.apply(init_weights)
        self._actor_target.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = optim.AdamW(
            self._actor.parameters(), lr=actor_learning_rate, amsgrad=True
        )

        # critic network
        self._critics = NpletsCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=critic_learning_rate,
            network_type=critic_network_type,
            init_fn=init_weights,
            num_critics=num_critic_network,
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
        # critic learning
        report = self._critic_learn_batch(batch)
        # actor learning
        report.update(self._actor_learn_batch(batch))

        self._critics.update_target_networks(self._critic_soft_update_tau)
        update_target_network(
            self._actor_target, self._actor, self._actor_soft_update_tau
        )
        return report

    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        # we would like to maximize Q(s, a)
        action_batch = self._actor(batch.state)
        q_values = self._critics.get_q_values(
            state_batch=batch.state, action_batch=action_batch, target=False
        )
        loss = -q_values.mean()
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return {"actor_loss": loss.mean().item()}

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        def target_fn(critic):
            return critic.get_batch_action_value(
                state_batch=batch.state,
                action_batch=batch.action,
            )

        with torch.no_grad():
            # Compute the Bellman Target
            next_action = self._actor_target(batch.next_state)
            next_q = self._critics.get_q_values(
                state_batch=batch.next_state, action_batch=next_action, target=True
            )
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.done)
            ) + batch.reward  # (batch_size), r + gamma * Q(s', a from actor network)

        losses = self._critics.optimize(target_fn, expected_state_action_values)
        return {"critic_loss": sum(losses) / len(losses)}
