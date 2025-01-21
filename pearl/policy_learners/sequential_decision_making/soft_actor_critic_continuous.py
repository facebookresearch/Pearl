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
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    GaussianActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
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
from pearl.utils.instantiations.spaces.box import BoxSpace
from torch import nn, optim


class ContinuousSoftActorCritic(ActorCriticBase):
    """
    Soft Actor Critic Policy Learner.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        state_dim: int | None = None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        history_summarization_learning_rate: float = 1e-3,
        actor_network_type: type[ActorNetwork] = GaussianActorNetwork,
        critic_network_type: type[QValueNetwork] = VanillaQValueNetwork,
        critic_soft_update_tau: float = 0.005,
        exploration_module: ExplorationModule | None = None,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 256,
        entropy_coef: float = 0.2,
        entropy_autotune: bool = True,
        action_representation_module: ActionRepresentationModule | None = None,
        actor_network_instance: ActorNetwork | None = None,
        critic_network_instance: QValueNetwork | nn.Module | None = None,
        actor_optimizer: Optional[optim.Optimizer] = None,
        critic_optimizer: Optional[optim.Optimizer] = None,
        history_summarization_optimizer: Optional[optim.Optimizer] = None,
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
            actor_soft_update_tau=0.0,
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
            is_action_continuous=True,
            on_policy=False,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            history_summarization_optimizer=history_summarization_optimizer,
        )

        self._entropy_autotune = entropy_autotune
        if entropy_autotune:
            # initialize the entropy coefficient to 0
            self.register_parameter(
                "_log_entropy",
                torch.nn.Parameter(torch.zeros(1, requires_grad=True)),
            )
            self._entropy_optimizer: torch.optim.Optimizer = optim.AdamW(
                # pyre-fixme[6]: For 1st argument expected `Union[Iterable[Dict[str,
                #  Any]], Iterable[Tuple[str, Tensor]], Iterable[Tensor]]` but got
                #  `List[Union[Module, Tensor]]`.
                [self._log_entropy],
                lr=self._critic_learning_rate,
                amsgrad=True,
            )
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            self.register_buffer("_entropy_coef", torch.exp(self._log_entropy).detach())
            assert isinstance(action_space, BoxSpace)
            self.register_buffer(
                "_target_entropy", -torch.tensor(action_space.shape[0])
            )
        else:
            self.register_buffer("_entropy_coef", torch.tensor(entropy_coef))
        self._action_batch_log_prob_cache: torch.Tensor = torch.tensor(0.0)

    def learn_batch(self, batch: TransitionBatch) -> dict[str, Any]:
        actor_critic_loss = super().learn_batch(batch)

        if self._entropy_autotune:
            entropy_optimizer_loss = (
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Union[Module, Tensor]`.
                -torch.exp(self._log_entropy)
                * (self._action_batch_log_prob_cache + self._target_entropy).detach()
            ).mean()

            self._entropy_optimizer.zero_grad()
            entropy_optimizer_loss.backward()
            self._entropy_optimizer.step()

            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Union[Module, Tensor]`.
            self._entropy_coef = torch.exp(self._log_entropy).detach()
            actor_critic_loss = {
                **actor_critic_loss,
                **{"entropy_coef": entropy_optimizer_loss},
            }

        return actor_critic_loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        reward_batch = batch.reward  # shape: (batch_size)
        terminated_batch = batch.terminated  # shape: (batch_size)

        if terminated_batch is not None:
            expected_state_action_values = (
                self._get_next_state_expected_values(batch)
                * self._discount_factor
                * (1 - terminated_batch.float())
            ) + reward_batch  # shape of expected_state_action_values: (batch_size)
        else:
            raise AssertionError("terminated_batch should not be None")

        loss, _, _ = twin_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            # pyre-fixme
            critic=self._critic,
        )

        return loss

    @torch.no_grad()
    def _get_next_state_expected_values(self, batch: TransitionBatch) -> torch.Tensor:
        next_state_batch = batch.next_state  # shape: (batch_size x state_dim)

        # shape of next_action_batch: (batch_size, action_dim)
        # shape of next_action_log_prob: (batch_size, 1)
        (
            next_action_batch,
            next_action_batch_log_prob,
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        ) = self._actor.sample_action(next_state_batch, get_log_prob=True)

        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        next_q1, next_q2 = self._critic_target.get_q_values(
            state_batch=next_state_batch,
            action_batch=next_action_batch,
        )

        # clipped double q-learning (reduce overestimation bias)
        next_q = torch.minimum(next_q1, next_q2)  # shape: (batch_size)
        next_state_action_values = next_q.unsqueeze(-1)  # shape: (batch_size x 1)

        # add entropy regularization
        next_state_action_values = next_state_action_values - (
            self._entropy_coef * next_action_batch_log_prob
        )
        # shape: (batch_size x 1)

        return next_state_action_values.view(-1)

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state  # shape: (batch_size x state_dim)

        # shape of action_batch: (batch_size, action_dim)
        # shape of action_batch_log_prob: (batch_size, 1)
        (
            action_batch,
            action_batch_log_prob,
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        ) = self._actor.sample_action(state_batch, get_log_prob=True)

        self._action_batch_log_prob_cache = action_batch_log_prob
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        q1, q2 = self._critic.get_q_values(
            state_batch=state_batch, action_batch=action_batch
        )  # shape: (batch_size, 1)

        # clipped double q learning (reduce overestimation bias)
        q = torch.minimum(q1, q2)  # shape: (batch_size)
        state_action_values = q.unsqueeze(-1)  # shape: (batch_size x 1)

        loss = (self._entropy_coef * action_batch_log_prob - state_action_values).mean()

        return loss

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two ContinuousSoftActorCritic instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, ContinuousSoftActorCritic):
            differences.append("other is not an instance of ContinuousSoftActorCritic")
        else:
            # Compare attributes specific to ContinuousSoftActorCritic
            if self._entropy_autotune != other._entropy_autotune:
                differences.append(
                    f"_entropy_autotune is different: {self._entropy_autotune} "
                    + f"vs {other._entropy_autotune}"
                )
            if not torch.allclose(self._entropy_coef, other._entropy_coef):
                differences.append(
                    f"_entropy_coef is different: {self._entropy_coef} vs {other._entropy_coef}"
                )
            if not torch.allclose(self._target_entropy, other._target_entropy):
                differences.append(
                    f"_target_entropy is different: {self._target_entropy} "
                    + f"vs {other._target_entropy}"
                )

        return "\n".join(differences)
