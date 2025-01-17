# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

from typing import Any, Dict, List, Optional

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaActorNetwork,
)

from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
    VanillaQValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
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
from pearl.utils.instantiations.spaces.discrete import DiscreteSpace

from torch import nn, optim


# Currently available actions is not used. Needs to be updated once we know the input
# structure of production stack on this param.


# TODO: to make things easier with a single optimizer, we need to polish this method.
class SoftActorCritic(ActorCriticBase):
    """
    Implementation of Soft Actor Critic Policy Learner for discrete action spaces.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        state_dim: int | None = None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        history_summarization_learning_rate: float = 1e-4,
        actor_network_type: type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: type[QValueNetwork] = VanillaQValueNetwork,
        critic_soft_update_tau: float = 0.005,
        exploration_module: ExplorationModule | None = None,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        entropy_coef: float = 0.2,
        entropy_autotune: bool = True,
        action_representation_module: ActionRepresentationModule | None = None,
        actor_network_instance: ActorNetwork | None = None,
        critic_network_instance: QValueNetwork | nn.Module | None = None,
        actor_optimizer: Optional[optim.Optimizer] = None,
        critic_optimizer: Optional[optim.Optimizer] = None,
        history_summarization_optimizer: Optional[optim.Optimizer] = None,
        target_entropy_scale: float = 0.89,
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
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=critic_soft_update_tau,
            use_twin_critic=True,
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else PropensityExploration()
            ),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=False,
            on_policy=False,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            history_summarization_optimizer=history_summarization_optimizer,
        )

        # This is needed to avoid actor softmax overflow issue.
        # Should not be left for users to choose.
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self._actor_optimizer, gamma=0.99
        )

        # TODO: implement learnable entropy coefficient
        self._entropy_autotune = entropy_autotune
        if entropy_autotune:
            # initialize the entropy coefficient to 0
            self.register_parameter(
                "_log_entropy",
                torch.nn.Parameter(torch.zeros(1, requires_grad=True)),
            )
            self._entropy_optimizer: torch.optim.Optimizer = optim.Adam(
                # pyre-fixme[6]: In call `optim.adam.Adam.__init__`, for 1st positional argument,
                # expected `Union[Iterable[Dict[str, typing.Any]], Iterable[Tuple[str, Tensor]],
                # Iterable[Tensor]]` but got `List[Union[Module, Tensor]]`.
                [self._log_entropy],
                lr=self._critic_learning_rate,
                eps=1e-4,
            )
            # pyre-fixme[6]: In call `optim.adam.Adam.__init__`, for 1st positional argument,
            # expected `Union[Iterable[Dict[str, typing.Any]], Iterable[Tuple[str, Tensor]],
            # Iterable[Tensor]]` but got `List[Union[Module, Tensor]]`.
            self.register_buffer("_entropy_coef", torch.exp(self._log_entropy).detach())
            assert isinstance(action_space, DiscreteSpace)
            self.register_buffer(
                "_target_entropy",
                -target_entropy_scale * torch.log(1.0 / torch.tensor(action_space.n)),
            )
        else:
            self.register_buffer("_entropy_coef", torch.tensor(entropy_coef))

    # sac uses a learning rate scheduler specifically
    def reset(self, action_space: ActionSpace) -> None:
        # pyre-fixme[16]: `SoftActorCritic` has no attribute `_action_space`.
        self._action_space = action_space
        self.scheduler.step()

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        actor_critic_loss = super().learn_batch(batch)

        if self._entropy_autotune:
            entropy = (
                # pyre-fixme[29]: `Union[BoundMethod[typing.Callable(torch._C.TensorBase.__mul__)
                # [[Named(self, torch._C.TensorBase), Named(other, Union[bool, complex, float,
                # int, torch._tensor.Tensor])], torch._tensor.Tensor], torch._tensor.Tensor],
                # nn.modules.module.Module, torch._tensor.Tensor]` is not a function.
                -(self._action_probs_cache * self._action_log_probs_cache).sum(1).mean()
            )
            entropy_optimizer_loss = (
                # pyre-fixme[6]: In call `torch._C._VariableFunctions.exp`,
                # for 1st positional argument, expected `Tensor` but got `Union[Module, Tensor]`.
                torch.exp(self._log_entropy) * (entropy - self._target_entropy).detach()
            )

            self._entropy_optimizer.zero_grad()
            entropy_optimizer_loss.backward()
            self._entropy_optimizer.step()
            # pyre-fixme[6]: In call `torch._C._VariableFunctions.exp`,
            # for 1st positional argument, expected `Tensor` but got `Union[Module, Tensor]`.
            self._entropy_coef = torch.exp(self._log_entropy).detach()
            actor_critic_loss = {
                **actor_critic_loss,
                **{"entropy_coef": entropy_optimizer_loss},
            }

        return actor_critic_loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        reward_batch = batch.reward  # (batch_size)
        terminated_batch = batch.terminated  # (batch_size)

        assert terminated_batch is not None
        expected_state_action_values = (
            self._get_next_state_expected_values(batch)
            * self._discount_factor
            * (1 - terminated_batch.float())
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        assert isinstance(self._critic, TwinCritic)
        loss, _, _ = twin_critic_action_value_loss(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            critic=self._critic,
        )

        return loss

    @torch.no_grad()
    def _get_next_state_expected_values(self, batch: TransitionBatch) -> torch.Tensor:
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_available_actions_batch = (
            batch.next_available_actions
        )  # (batch_size x action_space_size x action_dim)
        next_unavailable_actions_mask_batch = (
            batch.next_unavailable_actions_mask
        )  # (batch_size x action_space_size)

        assert next_state_batch is not None
        assert next_available_actions_batch is not None
        # get q values of (states, all actions) from twin critics
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        next_q1, next_q2 = self._critic_target.get_q_values(
            state_batch=next_state_batch,
            action_batch=next_available_actions_batch,
        )  # (batch_size, action_space_size), (batch_size, action_space_size)

        # clipped double q-learning (reduce overestimation bias)
        next_q = torch.minimum(next_q1, next_q2)

        # random ensemble distillation (reduce overestimation bias)
        # random_index = torch.randint(0, 2, (1,)).item()
        # next_q = next_q1 if random_index == 0 else next_q2

        # Make sure that unavailable actions' Q values are assigned to 0.0
        # since we are calculating expectation

        if next_unavailable_actions_mask_batch is not None:
            next_q[next_unavailable_actions_mask_batch] = 0.0

        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        next_state_policy_dist = self._actor.get_policy_distribution(
            state_batch=next_state_batch,
            available_actions=next_available_actions_batch,
            unavailable_actions_mask=next_unavailable_actions_mask_batch,
        )  # (batch_size x action_space_size)

        # Entropy Regularization
        next_q = (
            next_q - self._entropy_coef * torch.log(next_state_policy_dist + 1e-8)
        ) * next_state_policy_dist  # (batch_size x action_space_size)

        return next_q.sum(dim=1)

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        state_batch = batch.state  # (batch_size x state_dim)

        available_actions = (
            batch.curr_available_actions
        )  # (batch_size x action_space_size x action_dim)

        # get q values of (states, all actions) from twin critics
        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        q1, q2 = self._critic.get_q_values(
            state_batch=state_batch,
            action_batch=available_actions,
        )  # (batch_size, action_space_size), (batch_size, action_space_size)
        # clipped double q learning (reduce overestimation bias)
        q = torch.minimum(q1, q2)

        unavailable_actions_mask = (
            batch.curr_unavailable_actions_mask
        )  # (batch_size x action_space_size)

        # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
        new_policy_dist = self._actor.get_policy_distribution(
            state_batch=state_batch,
            available_actions=available_actions,
            unavailable_actions_mask=unavailable_actions_mask,
        )  # (batch_size x action_space_size)
        # pyre-fixme[16]: `SoftActorCritic` has no attribute `_action_probs_cache`.
        self._action_probs_cache = new_policy_dist
        # pyre-fixme[16]: `SoftActorCritic` has no attribute `_action_log_probs_cache`.
        self._action_log_probs_cache = torch.log(new_policy_dist + 1e-8)
        if unavailable_actions_mask is not None:
            q[unavailable_actions_mask] = 0.0

        loss = (
            # pyre-fixmeUnsupported operand [58]: `*` is not supported for operand types
            # `torch._tensor.Tensor` and `Union[nn.modules.module.Module, torch._tensor.Tensor]`.
            new_policy_dist * (self._entropy_coef * self._action_log_probs_cache - q)
        ).mean()

        return loss

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two SoftActorCritic instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        differences.extend(super().compare(other))

        if not isinstance(other, SoftActorCritic):
            differences.append("other is not an instance of SoftActorCritic")
        else:
            # Compare attributes specific to SoftActorCritic
            if self._entropy_autotune != other._entropy_autotune:
                differences.append(
                    f"_entropy_autotune is different: {self._entropy_autotune} vs {other._entropy_autotune}"
                )
            if not torch.allclose(self._entropy_coef, other._entropy_coef):
                differences.append(
                    f"_entropy_coef is different: {self._entropy_coef} vs {other._entropy_coef}"
                )
            if not torch.allclose(self._target_entropy, other._target_entropy):
                differences.append(
                    f"_target_entropy is different: {self._target_entropy} vs {other._target_entropy}"
                )

        return "\n".join(differences)
