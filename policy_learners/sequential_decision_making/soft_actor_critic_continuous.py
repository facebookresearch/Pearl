from typing import Any, Dict, Iterable, Type

import torch
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetworkType,
    GaussianActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    OffPolicyActorCritic,
)

from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.device import get_pearl_device
from pearl.utils.functional_utils.learning.nn_learning_utils import (
    optimize_twin_critics_towards_target,
)
from torch import optim


class ContinuousSoftActorCritic(OffPolicyActorCritic):
    """
    Soft Actor Critic Policy Learner.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_dims: Iterable[int],
        # pyre-fixme[9]: exploration_module has type `ExplorationModule`; used as
        #  `None`.
        exploration_module: ExplorationModule = None,
        critic_learning_rate: float = 0.0001,
        actor_learning_rate: float = 0.0001,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        entropy_coef: float = 0.2,
        entropy_autotune: bool = True,
        critic_soft_update_tau: float = 0.005,
        actor_network_type: ActorNetworkType = GaussianActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
    ) -> None:
        super(ContinuousSoftActorCritic, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            critic_learning_rate=critic_learning_rate,
            actor_learning_rate=actor_learning_rate,
            exploration_module=exploration_module
            if exploration_module is not None
            else NoExploration(),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            critic_soft_update_tau=critic_soft_update_tau,
            is_action_continuous=True,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
        )

        device = get_pearl_device()
        self._entropy_autotune = entropy_autotune
        if entropy_autotune:
            # initialize the entropy coefficient to 0
            self._log_entropy = torch.zeros(  # pyre-ignore
                1, requires_grad=True, device=device
            )
            self._entropy_optimizer = optim.AdamW(  # pyre-ignore
                [self._log_entropy], lr=critic_learning_rate, amsgrad=True
            )
            self._entropy_coef = torch.exp(self._log_entropy).detach()  # pyre-ignore
            self._target_entropy = -torch.tensor(  # pyre-ignore
                action_space.shape[0]  # pyre-ignore
            )
        else:
            self._entropy_coef = torch.tensor(entropy_coef)

        self._rounds = 0

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        reward_batch = batch.reward  # shape: (batch_size)
        done_batch = batch.done  # shape: (batch_size)

        if done_batch is not None:
            expected_state_action_values = (
                self._get_next_state_expected_values(batch)
                * self._discount_factor
                * (1 - done_batch.float())
            ) + reward_batch  # shape of expected_state_action_values: (batch_size)
        else:
            raise AssertionError("done_batch should not be None")

        loss_critic_update = optimize_twin_critics_towards_target(
            twin_critic=self._twin_critics,
            optimizer=self._critic_optimizer,
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target=expected_state_action_values,
        )

        return loss_critic_update

    @torch.no_grad()
    def _get_next_state_expected_values(self, batch: TransitionBatch) -> torch.Tensor:
        next_state_batch = batch.next_state  # shape: (batch_size x state_dim)

        # shape of next_action_batch: (batch_size, action_dim)
        # shape of next_action_log_prob: (batch_size, 1)
        (
            next_action_batch,
            next_action_batch_log_prob,
        ) = self._actor.sample_action_and_get_log_prob(next_state_batch)

        next_q1, next_q2 = self._targets_of_twin_critics.get_twin_critic_values(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            state_batch=next_state_batch,
            action_batch=next_action_batch,
        )

        # clipped double q-learning (reduce overestimation bias)
        next_q = torch.minimum(next_q1, next_q2)  # shape: (batch_size)
        next_state_action_values = next_q.view(
            self.batch_size, 1
        )  # shape: (batch_size x 1)

        # add entropy regularization
        next_state_action_values = next_state_action_values - (
            self._entropy_coef * next_action_batch_log_prob
        )
        # shape: (batch_size x 1)

        return next_state_action_values.view(-1)

    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        state_batch = batch.state  # shape: (batch_size x state_dim)

        # shape of action_batch: (batch_size, action_dim)
        # shape of action_batch_log_prob: (batch_size, 1)
        (
            action_batch,
            action_batch_log_prob,
        ) = self._actor.sample_action_and_get_log_prob(state_batch)

        q1, q2 = self._twin_critics.get_twin_critic_values(
            state_batch=state_batch, action_batch=action_batch
        )  # shape: (batch_size, 1)

        # clipped double q learning (reduce overestimation bias)
        q = torch.minimum(q1, q2)  # shape: (batch_size)
        state_action_values = q.view((self.batch_size, 1))  # shape: (batch_size x 1)

        actor_loss = (
            self._entropy_coef * action_batch_log_prob - state_action_values
        ).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        if self._entropy_autotune:
            with torch.no_grad():
                _, action_batch_log_prob = self._actor.sample_action_and_get_log_prob(
                    state_batch
                )

            entropy_optimizer_loss = (
                -torch.exp(self._log_entropy)
                * (action_batch_log_prob + self._target_entropy)
            ).mean()

            self._entropy_optimizer.zero_grad()
            entropy_optimizer_loss.backward()
            self._entropy_optimizer.step()

            self._entropy_coef = torch.exp(self._log_entropy).detach()

        return {"actor_loss": actor_loss.item()}
