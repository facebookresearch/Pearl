from typing import Any, Dict, Iterable, Optional, Type

import torch

from pearl.api.action_space import ActionSpace
from pearl.history_summarization_modules.history_summarization_module import (
    HistorySummarizationModule,
)
from pearl.neural_networks.common.utils import update_target_networks

from pearl.neural_networks.common.value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
    VanillaValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetworkType,
    VanillaActorNetwork,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
    twin_critic_action_value_update,
)

from pearl.replay_buffers.transition import TransitionBatch
from torch import optim


class ImplicitQLearning(ActorCriticBase):
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
        action_space: ActionSpace,
        actor_hidden_dims: Iterable[int],
        critic_hidden_dims: Iterable[int],
        state_value_critic_hidden_dims: Iterable[int],
        exploration_module: Optional[ExplorationModule] = None,
        state_value_learning_rate: float = 1e-3,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: ActorNetworkType = VanillaActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        # pyre-fixme
        state_value_network_type=VanillaValueNetwork,
        critic_soft_update_tau: float = 0.05,
        discount_factor: float = 0.99,
        training_rounds: int = 5,
        batch_size: int = 128,
        expectile: float = 0.5,
        temperature_advantage_weighted_regression: float = 0.5,
        advantage_clamp: float = 100.0,
    ) -> None:

        super(ImplicitQLearning, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=False,
            use_critic_target=True,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=critic_soft_update_tau,
            use_twin_critic=True,
            exploration_module=exploration_module
            if exploration_module is not None
            else NoExploration(),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=False,
            on_policy=False,
        )

        self._expectile = expectile
        self._temperature_advantage_weighted_regression = (
            temperature_advantage_weighted_regression
        )
        self._advantage_clamp = advantage_clamp
        # iql uses both q and v approximators
        # pyre-fixme
        self._value_network = state_value_network_type(
            input_dim=state_dim,
            hidden_dims=state_value_critic_hidden_dims,
            output_dim=1,
        )
        self._value_network_optimizer = optim.AdamW(
            self._value_network.parameters(), lr=state_value_learning_rate, amsgrad=True
        )

    def set_history_summarization_module(
        self, value: HistorySummarizationModule
    ) -> None:
        self._actor_optimizer.add_param_group({"params": value.parameters()})
        self._critic_optimizer.add_param_group({"params": value.parameters()})
        self._value_network_optimizer.add_param_group({"params": value.parameters()})
        self._history_summarization_module = value

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        value_loss = self._value_learn_batch(batch)  # update value network
        actor_loss = self._actor_learn_batch(batch)  # update actor network
        critic_loss = self._critic_learn_batch(batch)  # update critic networks

        # update critic and target Twin networks;
        update_target_networks(
            self._critic_target._critic_networks_combined,
            self._critic._critic_networks_combined,
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
            q1, q2 = self._critic_target.get_q_values(batch.state, batch.action)
            # clipped double q-learning for dealing with overestimation bias
            # target_q = torch.minimum(q1, q2)
            random_index = torch.randint(0, 2, (1,)).item()
            target_q = (
                q1 if random_index == 0 else q2
            )  # random ensemble distillation, an alternative to clipped double q-learning

            value_batch = self._value_network(batch.state).view(-1)
            advantage = torch.exp(
                (target_q - value_batch)
                * self._temperature_advantage_weighted_regression
            )
            advantage = torch.clamp(advantage, max=self._advantage_clamp)

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
            values_next_states = self._value_network(batch.next_state).view(
                -1
            )  # shape: (batch_size)
            # To do: add interface to vanilla value networks
            # like vanilla q value networks using the 'get' function

            # compute targets for batch of (state, action, next_state)
            target = (
                values_next_states * self._discount_factor * (1 - batch.done.float())
            ) + batch.reward  # shape: (batch_size);  target y = r + gamma * V(s')
        # update twin critics towards target
        loss_critic_update = twin_critic_action_value_update(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=target,
            optimizer=self._critic_optimizer,
            # pyre-fixme
            critic=self._critic,
        )
        return loss_critic_update

    def _value_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        with torch.no_grad():
            q1, q2 = self._critic_target.get_q_values(batch.state, batch.action)
            # clipped double q-learning for dealing with overestimation bias
            # target_q = torch.minimum(q1, q2)
            random_index = torch.randint(0, 2, (1,)).item()
            target_q = (
                q1 if random_index == 0 else q2
            )  # random ensemble distillation, an alternative to clipped double q-learning

        value_batch = self._value_network(batch.state).view(
            -1
        )  # note the change in loss function from a mean square loss to an expectile loss
        # print(value_batch)
        # print(target_q)
        loss_value_network = self._expectile_loss(target_q - value_batch).mean()
        self._value_network_optimizer.zero_grad()
        loss_value_network.backward()
        self._value_network_optimizer.step()
        return loss_value_network.mean().item()

    # we do not expect this method to be reused in different algorithms, so it is defined here
    # To Do: add a utils method separately if needed in future for other algorithms to reuse
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _expectile_loss(self, input_loss):
        """
        Expectile loss applies an asymmetric weight
        to the input loss function parameterized by self._expectile.
        """
        weight = torch.where(input_loss > 0, self._expectile, (1 - self._expectile))
        return weight * (input_loss**2)
