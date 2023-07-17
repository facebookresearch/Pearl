from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from pearl.api.action_space import ActionSpace
from pearl.core.common.neural_networks.utils import init_weights, update_target_network
from pearl.core.common.neural_networks.value_networks import (
    StateActionValueNetworkType,
    VanillaStateActionValueNetwork,
)
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.core.sequential_decision_making.neural_networks.actor_networks import (
    ActorNetworkType,
    VanillaActorNetwork,
)
from pearl.core.sequential_decision_making.policy_learners.policy_gradient import (
    PolicyGradient,
)
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
        critic_network_type: StateActionValueNetworkType = VanillaStateActionValueNetwork,
    ) -> None:
        super(SoftActorCritic, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            exploration_module=exploration_module,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
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

        self._critic_1 = make_specified_critic_network()
        self._critic_1_target = make_specified_critic_network()
        self._critic_1.apply(init_weights)
        self._critic_1_target.load_state_dict(self._critic_1.state_dict())

        self._critic_2 = make_specified_critic_network()
        self._critic_2_target = make_specified_critic_network()
        self._critic_2.apply(init_weights)
        self._critic_2_target.load_state_dict(self._critic_2.state_dict())

        self._critic_1_optimizer = optim.AdamW(
            self._critic_1.parameters(), lr=learning_rate, amsgrad=True
        )
        self._critic_2_optimizer = optim.AdamW(
            self._critic_2.parameters(), lr=learning_rate, amsgrad=True
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

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space
        self.scheduler.step()

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        # critic learning
        self._critic_learn_batch(batch)
        # actor learning
        self._actor_learn_batch(batch)

        self._rounds += 1

        update_target_network(
            self._critic_1_target, self._critic_1, self._soft_update_tau
        )
        update_target_network(
            self._critic_2_target, self._critic_2, self._soft_update_tau
        )

        return {}

    def _critic_learn_batch(self, batch: TransitionBatch) -> None:
        state_batch = batch.state  # (batch_size x state_dim)
        action_batch = batch.action  # (batch_size x action_dim)
        reward_batch = batch.reward  # (batch_size)
        done_batch = batch.done  # (batch_size)

        batch_size = state_batch.shape[0]
        # sanity check they have same batch_size
        assert state_batch.shape[0] == batch_size
        assert action_batch.shape[0] == batch_size
        assert reward_batch.shape[0] == batch_size
        assert done_batch.shape[0] == batch_size

        state_action_values_1 = self._critic_1.get_batch_action_value(
            state_batch=state_batch,
            action_batch=action_batch,
            curr_available_actions_batch=batch.curr_available_actions,
        )  # (batch_size)

        state_action_values_2 = self._critic_2.get_batch_action_value(
            state_batch=state_batch,
            action_batch=action_batch,
            curr_available_actions_batch=batch.curr_available_actions,
        )  # (batch_size)

        expected_state_action_values = (
            self._get_next_state_expected_values(batch)
            * self._discount_factor
            * (1 - done_batch)
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        criterion = torch.nn.MSELoss()

        critic_1_loss = criterion(state_action_values_1, expected_state_action_values)
        self._critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self._critic_1_optimizer.step()

        critic_2_loss = criterion(state_action_values_2, expected_state_action_values)
        self._critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self._critic_2_optimizer.step()

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

        next_state_action_values_1 = self._critic_1_target.get_batch_action_value(
            next_state_batch_repeated,
            next_available_actions_batch,
        ).view(
            (self.batch_size, -1)
        )  # (batch_size x action_space_size)
        next_state_action_values_2 = self._critic_2_target.get_batch_action_value(
            next_state_batch_repeated,
            next_available_actions_batch,
        ).view(
            (self.batch_size, -1)
        )  # (batch_size x action_space_size)
        next_state_action_values = torch.min(
            next_state_action_values_1, next_state_action_values_2
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

        batch_size = state_batch.shape[0]
        # sanity check they have same batch_size
        assert state_batch.shape[0] == batch_size

        # TODO: assumes all current actions are available. Needs to fix.
        action_space = (
            F.one_hot(torch.arange(0, self._action_space.n))
            .unsqueeze(0)
            .repeat(self.batch_size, 1, 1)
        )  # (batch_size x action_space_size x action_dim)

        new_policy_dist = self._actor(state_batch)  # (batch_size x action_space_size)
        state_batch_repeated = torch.repeat_interleave(
            state_batch.unsqueeze(1), self._action_space.n, dim=1
        )  # (batch_size x action_space_size x state_dim)

        state_action_values_1 = self._critic_1.get_batch_action_value(
            state_batch_repeated, action_space
        ).view(
            (self.batch_size, self._action_space.n)
        )  # (batch_size x action_space_size)
        state_action_values_2 = self._critic_2.get_batch_action_value(
            state_batch_repeated, action_space
        ).view(
            (self.batch_size, self._action_space.n)
        )  # (batch_size x action_space_size)
        state_action_values = torch.min(
            state_action_values_1, state_action_values_2
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
