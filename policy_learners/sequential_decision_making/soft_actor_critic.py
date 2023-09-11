from typing import Any, Dict, Iterable, Type

import torch
import torch.nn.functional as F
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.utils import init_weights, update_target_networks
from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetworkType,
    VanillaActorNetwork,
)

from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)

from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.policy_gradient import (
    PolicyGradient,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import optim


# TODO: Only support discrete action space problems for now and assumes Gym action space.
# Currently available actions is not used. Needs to be updated once we know the input structure
# of production stack on this param.

# TODO: to make things easier with a single optimizer, we need to further polish this method.
class SoftActorCritic(PolicyGradient):
    """
    Soft Actor Critic Policy Learner.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_dims: Iterable[int],
        exploration_module: ExplorationModule = None,
        learning_rate: float = 0.0001,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        entropy_coef: float = 0.2,
        soft_update_tau: float = 0.005,
        actor_network_type: ActorNetworkType = VanillaActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
    ) -> None:
        super(SoftActorCritic, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            exploration_module=exploration_module,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            network_type=actor_network_type,
            on_policy=False,
        )

        # TODO: Assumes Gym interface, fix it.
        def make_specified_critic_network():
            return critic_network_type(
                state_dim=state_dim,
                action_dim=action_space.n,
                hidden_dims=hidden_dims,
                output_dim=1,
            )

        # twin critic: using two separate critic networks to reduce overestimation bias
        # optimizers of two critics are alredy initialized in TwinCritic
        self._twin_critics = TwinCritic(
            state_dim=state_dim,
            action_dim=action_space.n,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            network_type=critic_network_type,
            init_fn=init_weights,
        )

        # target networks of twin critics
        self._targets_of_twin_critics = TwinCritic(
            state_dim=state_dim,
            action_dim=action_space.n,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            network_type=critic_network_type,
            init_fn=init_weights,
        )

        # target networks are initialized to parameters of the source network (tau is set to 1)
        update_target_networks(
            self._targets_of_twin_critics._critic_networks_combined,
            self._twin_critics._critic_networks_combined,
            tau=1,
        )

        # This is needed to avoid actor softmax overflow issue.
        # Should not be left for users to choose.
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self._actor_optimizer, gamma=0.99
        )

        # TODO: implement learnable entropy coefficient
        self._entropy_coef = entropy_coef
        self._rounds = 0

        self._soft_update_tau = soft_update_tau
        self._training_rounds = training_rounds

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space
        self.scheduler.step()

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        self._critic_learn_batch(batch)  # update critic
        self._actor_learn_batch(batch)  # update actor

        self._rounds += 1

        # update targets of twin critics using soft updates
        update_target_networks(
            self._targets_of_twin_critics._critic_networks_combined,
            self._twin_critics._critic_networks_combined,
            self._soft_update_tau,
        )
        return {}

    def _critic_learn_batch(self, batch: TransitionBatch) -> None:

        reward_batch = batch.reward  # (batch_size)
        done_batch = batch.done  # (batch_size)

        expected_state_action_values = (
            self._get_next_state_expected_values(batch)
            * self._discount_factor
            * (1 - done_batch)
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        # self._critics.optimize(target_fn, expected_state_action_values)
        loss_critic_update = self._twin_critics.optimize_twin_critics_towards_target(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target=expected_state_action_values,
        )

        return loss_critic_update

    @torch.no_grad()
    def _get_next_state_expected_values(self, batch: TransitionBatch) -> torch.tensor:
        next_state_batch = batch.next_state  # (batch_size x state_dim)
        next_available_actions_batch = (
            batch.next_available_actions
        )  # (batch_size x action_space_size x action_dim)
        next_available_actions_mask_batch = (
            batch.next_available_actions_mask
        )  # (batch_size x action_space_size)

        next_state_batch_repeated = torch.repeat_interleave(
            next_state_batch.unsqueeze(1), self._action_space.n, dim=1
        )  # (batch_size x action_space_size x state_dim)

        # get q values of (states, all actions) from twin critics
        next_q1, next_q2 = self._targets_of_twin_critics.get_twin_critic_values(
            state_batch=next_state_batch_repeated,
            action_batch=next_available_actions_batch,
        )

        # clipped double q-learning (reduce overestimation bias)
        next_q = torch.minimum(next_q1, next_q2)

        # random ensemble distillation (reduce overestimation bias)
        # random_index = torch.randint(0, 2, (1,)).item()
        # next_q = next_q1 if random_index == 0 else next_q2

        next_state_action_values = next_q.view(
            self.batch_size, -1
        )  # (batch_size x action_space_size)

        # Make sure that unavailable actions' Q values are assigned to 0.0
        # since we are calculating expectation
        # TODO: evaluate if there's a better way to deal with variable action space
        # in actor-critic type of algorithms
        next_state_action_values[next_available_actions_mask_batch] = 0.0

        next_state_policy_dist = self._actor(
            next_state_batch
        )  # (batch_size x action_space_size)

        # Entropy Regularization
        next_state_action_values = (
            next_state_action_values
            - self._entropy_coef * torch.log(next_state_policy_dist)
        ) * next_state_policy_dist  # (batch_size x action_space_size)

        return next_state_action_values.sum(dim=1)

    def _actor_learn_batch(self, batch: TransitionBatch) -> None:
        state_batch = batch.state  # (batch_size x state_dim)

        # TODO: assumes all current actions are available. Needs to fix.
        action_space = (
            F.one_hot(torch.arange(0, self._action_space.n))
            .unsqueeze(0)
            .repeat(self.batch_size, 1, 1)
        ).to(
            self.device
        )  # (batch_size x action_space_size x action_dim)

        new_policy_dist = self._actor(state_batch)  # (batch_size x action_space_size)
        state_batch_repeated = torch.repeat_interleave(
            state_batch.unsqueeze(1), self._action_space.n, dim=1
        )  # (batch_size x action_space_size x state_dim)

        # get q values of (states, all actions) from twin critics
        q1, q2 = self._twin_critics.get_twin_critic_values(
            state_batch=state_batch_repeated, action_batch=action_space
        )
        # clipped double q learning (reduce overestimation bias)
        q = torch.minimum(q1, q2)

        # random ensemble distillation (reduce overestimation bias)
        # random_index = torch.randint(0, 2, (1,)).item()
        # q = q1 if random_index == 0 else q2

        state_action_values = q.view(
            (self.batch_size, self._action_space.n)
        )  # (batch_size x action_space_size)

        policy_loss = (
            (
                new_policy_dist
                * (
                    self._entropy_coef * torch.log(new_policy_dist)
                    - state_action_values
                )
            )
            .sum(dim=1)
            .mean()
        )

        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()
