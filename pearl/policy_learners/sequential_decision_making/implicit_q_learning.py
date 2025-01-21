# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, List, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)

from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.utils import update_target_networks
from pearl.neural_networks.common.value_networks import (
    ValueNetwork,
    VanillaValueNetwork,
)

from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaActorNetwork,
    VanillaContinuousActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)

from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)

from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.critic_utils import (
    twin_critic_action_value_loss,
)
from pearl.utils.module_utils import modules_have_similar_state_dict
from torch import optim


class ImplicitQLearning(ActorCriticBase):
    """
    Implementation of Implicit Q learning, an offline RL algorithm:
    https://arxiv.org/pdf/2110.06169.pdf.
    Author implementation in Jax: https://github.com/ikostrikov/implicit_q_learning

    Algorithm implementation:
     - perform value, crtic and actor updates sequentially
     - soft update target networks of twin critics using (tau)

    Note:
    This implementation uses twin critic to reduce
    overestimation bias. See TwinCritic class for implementation details.

    Args:
        expectile: a value between 0 and 1, for expectile regression
        temperature_advantage_weighted_regression: temperature parameter for advantage
        weighted regression; used to extract policy from trained value and critic networks.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        state_dim: int | None = None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        value_critic_hidden_dims: list[int] | None = None,
        exploration_module: ExplorationModule | None = None,
        actor_network_type: type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: type[QValueNetwork] = VanillaQValueNetwork,
        value_network_type: type[ValueNetwork] = VanillaValueNetwork,
        value_critic_learning_rate: float = 1e-3,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        history_summarization_learning_rate: float = 1e-3,
        critic_soft_update_tau: float = 0.05,
        discount_factor: float = 0.99,
        training_rounds: int = 5,
        batch_size: int = 128,
        expectile: float = 0.5,
        temperature_advantage_weighted_regression: float = 0.5,
        advantage_clamp: float = 100.0,
        action_representation_module: ActionRepresentationModule | None = None,
        actor_network_instance: ActorNetwork | None = None,
        critic_network_instance: QValueNetwork | None = None,
        value_network_instance: ValueNetwork | None = None,
        actor_optimizer: Optional[optim.Optimizer] = None,
        critic_optimizer: Optional[optim.Optimizer] = None,
        history_summarization_optimizer: Optional[optim.Optimizer] = None,
        value_optimizer: Optional[optim.Optimizer] = None,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            history_summarization_learning_rate=history_summarization_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=False,
            use_critic_target=True,
            critic_soft_update_tau=critic_soft_update_tau,
            use_twin_critic=True,
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else NoExploration()
            ),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=action_space.is_continuous,  # inferred from the action space
            on_policy=False,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            history_summarization_optimizer=history_summarization_optimizer,
        )

        self._expectile = expectile
        self._is_action_continuous: bool = action_space.is_continuous

        self._temperature_advantage_weighted_regression = (
            temperature_advantage_weighted_regression
        )
        self._advantage_clamp = advantage_clamp
        # iql uses both q and v approximators
        if value_network_instance is not None:
            self._value_network = value_network_instance
        else:
            assert state_dim is not None
            assert value_critic_hidden_dims is not None
            self._value_network: ValueNetwork = value_network_type(
                input_dim=state_dim,
                hidden_dims=value_critic_hidden_dims,
                output_dim=1,
            )
        if value_optimizer is not None:
            self._value_network_optimizer: optim.Optimizer = value_optimizer
        else:
            self._value_network_optimizer = optim.AdamW(
                self._value_network.parameters(),
                lr=value_critic_learning_rate,
                amsgrad=True,
            )

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        value_loss = self._value_loss(batch)
        critic_loss = self._critic_loss(batch)
        actor_loss = self._actor_loss(batch)
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `zero_grad`.
        self._history_summarization_optimizer.zero_grad()
        self._value_network_optimizer.zero_grad()
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        loss = value_loss + critic_loss + actor_loss
        loss.backward()
        self._value_network_optimizer.step()
        self._actor_optimizer.step()
        self._critic_optimizer.step()
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `step`.
        self._history_summarization_optimizer.step()
        # update critic and target Twin networks;
        update_target_networks(
            # pyre-fixme[6]: For 1st argument expected `Union[List[Module],
            #  ModuleList]` but got `Union[Tensor, Module]`.
            self._critic_target._critic_networks_combined,
            # pyre-fixme[6]: For 2nd argument expected `Union[List[Module],
            #  ModuleList]` but got `Union[Tensor, Module]`.
            self._critic._critic_networks_combined,
            self._critic_soft_update_tau,
        )

        return {
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def _value_loss(self, batch: TransitionBatch) -> torch.Tensor:
        with torch.no_grad():
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            q1, q2 = self._critic_target.get_q_values(batch.state, batch.action)
            # random ensemble distillation.
            random_index = torch.randint(0, 2, (1,)).item()
            target_q = q1 if random_index == 0 else q2  # shape: (batch_size)

        value_batch = self._value_network(batch.state).view(-1)  # shape: (batch_size)

        # note the change in loss function from a mean square loss to an expectile loss
        loss = self._expectile_loss(target_q - value_batch).mean()
        return loss

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Performs policy extraction using advantage weighted regression
        """
        with torch.no_grad():
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            q1, q2 = self._critic_target.get_q_values(batch.state, batch.action)
            # random ensemble distillation.
            random_index = torch.randint(0, 2, (1,)).item()
            target_q = q1 if random_index == 0 else q2  # shape: (batch_size)

            value_batch = self._value_network(batch.state).view(-1)
            # shape: (batch_size)

            advantage = torch.exp(
                (target_q - value_batch)
                * self._temperature_advantage_weighted_regression
            )  # shape: (batch_size)
            advantage = torch.clamp(advantage, max=self._advantage_clamp)

        # TODO: replace VanillaContinuousActorNetwork by a base interface
        # covering all deterministic actors
        if isinstance(self._actor, VanillaContinuousActorNetwork):
            # mean square error between the actor network output and action batch
            loss = (
                (self._actor.sample_action(batch.state) - batch.action)
                .pow(2)
                .mean(dim=1)
            )  # shape: (batch_size)

            # advantage weighted regression loss for training deterministic actors
            actor_loss = (advantage * loss).mean()

        else:
            if self._is_action_continuous:
                # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                log_action_probabilities = self._actor.get_log_probability(
                    batch.state, batch.action
                ).view(-1)
                # shape: (batch_size)

            else:
                action_probabilities = self._actor(batch.state)
                # shape: (batch_size, action_space_size)

                # one_hot to action indices
                action_idx = torch.argmax(batch.action, dim=1).unsqueeze(-1)

                # gather log probabilities of actions in the dataset
                log_action_probabilities = torch.log(
                    torch.gather(action_probabilities, 1, action_idx).view(-1)
                )

            # advantage weighted regression for stochastic actors
            actor_loss = -(advantage * log_action_probabilities).mean()

        return actor_loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        with torch.no_grad():
            # sample values of next states
            values_next_states = self._value_network(batch.next_state).view(-1)
            # shape: (batch_size)

            # TODO: add interface to vanilla value networks
            # like vanilla q value networks using the 'get' function

            # compute targets for batch of (state, action, next_state): target y = r + gamma * V(s')
            target = (
                values_next_states
                * self._discount_factor
                * (1 - batch.terminated.float())
            ) + batch.reward  # shape: (batch_size)

        assert isinstance(
            self._critic, TwinCritic
        ), "Critic in ImplicitQLearning should be TwinCritic"

        # update twin critics towards target
        loss, _, _ = twin_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=target,
            critic=self._critic,
        )
        return loss

    # we do not expect this method to be reused in different algorithms, so it is defined here
    # To Do: add a utils method separately if needed in future for other algorithms to reuse
    def _expectile_loss(self, input_loss: torch.Tensor) -> torch.Tensor:
        """
        Expectile loss applies an asymmetric weight
        to the input loss function parameterized by self._expectile.
        """
        weight = torch.where(input_loss > 0, self._expectile, (1 - self._expectile))
        return weight * (input_loss.pow(2))

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two ImplicitQLearning instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        # Inherit comparisons from the base class
        differences.append(super().compare(other))

        if not isinstance(other, ImplicitQLearning):
            differences.append("other is not an instance of ImplicitQLearning")
        else:
            # Compare attributes specific to ImplicitQLearning
            if self._expectile != other._expectile:
                differences.append(
                    f"_expectile is different: {self._expectile} vs {other._expectile}"
                )
            if (
                self._temperature_advantage_weighted_regression
                != other._temperature_advantage_weighted_regression
            ):
                differences.append(
                    "_temperature_advantage_weighted_regression is different: "
                    + f"{self._temperature_advantage_weighted_regression} "
                    + f"vs {other._temperature_advantage_weighted_regression}"
                )
            if self._advantage_clamp != other._advantage_clamp:
                differences.append(
                    f"_advantage_clamp is different: {self._advantage_clamp} "
                    + f"vs {other._advantage_clamp}"
                )

            # Compare value network using modules_have_similar_state_dict
            if (
                reason := modules_have_similar_state_dict(
                    self._value_network, other._value_network
                )
            ) != "":
                differences.append(f"_value_network is different: {reason}")

        return "\n".join(differences)
