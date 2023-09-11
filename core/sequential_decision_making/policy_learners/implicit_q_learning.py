from typing import Any, Dict, Iterable, Type

import torch

# from pearl.api.action import Action
from pearl.api.action_space import ActionSpace

# from pearl.api.state import SubjectiveState
from pearl.core.common.neural_networks.twin_critic import TwinCritic

from pearl.core.common.neural_networks.utils import init_weights, update_target_networks

from pearl.core.common.neural_networks.value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
    VanillaValueNetwork,
)
from pearl.core.common.policy_learners.exploration_module.no_exploration import (
    NoExploration,
)
from pearl.core.sequential_decision_making.neural_networks.actor_networks import (
    ActorNetworkType,
    VanillaActorNetwork,
)
from pearl.core.sequential_decision_making.policy_learners.policy_gradient import (
    PolicyGradient,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import optim


class ImplicitQLearning(PolicyGradient):
    """
    Implementation of Implicit Q learning, an offline RL algorithm: https://arxiv.org/pdf/2110.06169.pdf.
    Author implementation in Jax: https://github.com/ikostrikov/implicit_q_learning
    Implicit Q learning can be viewed as a policy gradient method, so based on PolicyGradient class.
    We use act and reset method from the PolicyGradient class.

    Algorithm implementation:
     - perform value, crtic and actor updates sequentially
     - soft update target networks of twin critics using (tau)

    Notes:
    1) Currently written for discrete action spaces. For continuous action spaces, we
    need to implement the reparameterization trick.
    2) This implementation uses twin critic (clipped double q learning) to reduce
    overestimation bias. See TwinCritic class for implementation details.

    Args: two noteworthy arguments:
        - expectile: a value between 0 and 1, for expectile regression
        - temperature_advantage_weighted_regression: temperature parameter for advantage
        weighted regression; used to extract policy from trained value and critic networks.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_space: ActionSpace,
        hidden_dims: Iterable[int],
        critic_learning_rate: float = 1e-3,
        actor_learning_rate: float = 1e-3,
        value_learning_rate: float = 1e-3,
        batch_size: int = 128,
        actor_network_type: ActorNetworkType = VanillaActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        training_rounds: int = 5,
        expectile: float = 0.8,
        critic_soft_update_tau: float = 0.05,
        discount_factor: float = 0.99,
        temperature_advantage_weighted_regression: float = 0.5,
    ) -> None:

        super(ImplicitQLearning, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            exploration_module=NoExploration(),  # since IQL is an offline learning method, exploration is not required
            learning_rate=actor_learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            network_type=actor_network_type,
            on_policy=False,
        )

        self._expectile = expectile
        self._discount_factor = discount_factor
        self._critic_soft_update_tau = critic_soft_update_tau
        self._training_rounds = training_rounds
        self._temperature_advantage_weighted_regression = (
            temperature_advantage_weighted_regression
        )

        # a single actor network
        self._actor = actor_network_type(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
        )
        self._actor.apply(init_weights)
        self._actor_optimizer = optim.AdamW(
            self._actor.parameters(), lr=actor_learning_rate, amsgrad=True
        )  # actor network's optimizer

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

        # targets of twin critics
        self._targets_of_twin_critics = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            learning_rate=critic_learning_rate,
            network_type=critic_network_type,
            init_fn=init_weights,
        )
        # for twin critic and corresponding target, there is a temptation to use deepcopy or clone instead.
        # this can however create problems in distibuted computing or because of shared buffers (when using batch norm for example)
        # params of targets of twin critics initialized to params of twin critics (tau is set to 1)
        update_target_networks(
            self._twin_critics._critic_networks_combined,
            self._targets_of_twin_critics._critic_networks_combined,
            tau=1,
        )

        # value network
        self._value_network = VanillaValueNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
        )
        self._value_network_optimizer = optim.AdamW(
            self._value_network.parameters(), lr=value_learning_rate, amsgrad=True
        )

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        value_loss = self._value_learn_batch(batch)  # update value network
        actor_loss = self._actor_learn_batch(batch)  # update actor network
        critic_loss = self._critic_learn_batch(batch)  # update critic networks

        # update critic and target Twin networks;
        update_target_networks(
            self._twin_critics._critic_networks_combined,
            self._targets_of_twin_critics._critic_networks_combined,
            self._critic_soft_update_tau,
        )

        return {
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }

    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        """
        performs policy extraction using advantage weighted regression
        """
        with torch.no_grad():
            q1, q2 = self._targets_of_twin_critics.get_twin_critic_values(
                batch.state, batch.action
            )
            # target_q = torch.minimum(q1, q2)  # clipped double q-learning for dealing with overestimation bias
            random_index = torch.randint(0, 2, (1,)).item()
            target_q = (
                q1 if random_index == 0 else q2
            )  # random ensemble distillation, an alternative to clipped double q-learning

            value_batch = self._value_network(batch.state)
            advantage = torch.exp(
                (target_q - value_batch)
                * self._temperature_advantage_weighted_regression
            )

        action_probabilities = self._actor(
            batch.state
        )  # shape: (batch_size, action_space_size)
        action_idx = torch.argmax(batch.action, dim=1).unsqueeze(
            -1
        )  # one_hot to action indices

        # gather log probabilities of actions in the dataset
        log_action_probabilities = torch.log(
            torch.gather(action_probabilities, 1, action_idx).view(-1)
        )
        # compute loss function for advantage weighted regression and take a step
        actor_loss = -(advantage * log_action_probabilities).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()
        return actor_loss.mean().item()

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        with torch.no_grad():
            # sample values of next states
            values_next_states = self._value_network(
                batch.next_state
            )  # To do: add interface to vanilla value networks like vanilla q value networks using the 'get' function

            # compute targets for batch of (state, action, next_state)
            target = (
                values_next_states * self._discount_factor * (1 - batch.done)
            ) + batch.reward  # shape: (batch_size);  target y = r + gamma * V(s')

        # update twin critics towards target
        loss_critic_update = self._twin_critics.optimize_twin_critics_towards_target(
            state_batch=batch.state, action_batch=batch.action, expected_target=target
        )
        return loss_critic_update

    def _value_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        with torch.no_grad():
            q1, q2 = self._targets_of_twin_critics.get_twin_critic_values(
                batch.state, batch.action
            )
            # target_q = torch.minimum(q1, q2)  # clipped double q-learning for dealing with overestimation bias
            random_index = torch.randint(0, 2, (1,)).item()
            target_q = (
                q1 if random_index == 0 else q2
            )  # random ensemble distillation, an alternative to clipped double q-learning

        value_batch = self._value_network(
            batch.state
        )  # note the change in loss function from a mean square loss to an expectile loss
        loss_value_network = self._expectile_loss(target_q - value_batch).mean()
        self._value_network_optimizer.zero_grad()
        loss_value_network.backward()
        self._value_network_optimizer.step()
        return loss_value_network.mean().item()

    # we do not expect this method to be reused in different algorithms, so it is defined here
    # To Do: add a utils method separately if needed in future for other algorithms to reuse
    def _expectile_loss(self, input_loss):
        """
        Expectile loss applies an asymmetric weight to the input loss function parameterized by self._expectile.
        """
        weight = torch.where(input_loss > 0, self._expectile, (1 - self._expectile))
        return weight * (input_loss**2)
