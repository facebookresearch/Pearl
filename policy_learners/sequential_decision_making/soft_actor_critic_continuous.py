# TODO: @hongbo, this file needs to be removed and we should add at most 20 lines of code to the
# soft_actor_critic.py file to support continuous action space.

from typing import Any, Dict, Iterable, Type

import torch
from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.common.utils import init_weights, update_target_network
from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetworkType,
    ContinousActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.policy_gradient import (
    PolicyGradient,
)
from pearl.replay_buffers.transition import TransitionBatch
from torch import optim


class ContinuousSoftActorCritic(PolicyGradient):
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
        actor_network_type: ActorNetworkType = ContinousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
    ) -> None:
        super(ContinuousSoftActorCritic, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            hidden_dims=hidden_dims,
            exploration_module=exploration_module,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            network_type=actor_network_type,
            on_policy=False,
            is_action_continuous=True,
        )

        # critic net
        def make_specified_critic_network():
            return critic_network_type(
                state_dim=state_dim,
                action_dim=1,
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

        self._entropy_coef = entropy_coef
        self._rounds = 0

        self._soft_update_tau = soft_update_tau
        self._training_rounds = training_rounds

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

        # ----- Q values TARGET -----
        expected_state_action_values = (
            self._get_next_state_expected_values(batch)
            * self._discount_factor
            * (1 - done_batch)
        ) + reward_batch  # (batch_size), r + gamma * V(s)

        # ----- Q values of Policy -----
        state_action_values_1 = self._critic_1.get_q_values(
            state_batch=state_batch,
            action_batch=action_batch,
            curr_available_actions_batch=batch.curr_available_actions,
        )  # (batch_size)

        state_action_values_2 = self._critic_2.get_q_values(
            state_batch=state_batch,
            action_batch=action_batch,
            curr_available_actions_batch=batch.curr_available_actions,
        )  # (batch_size)

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
        next_action_batch, next_action_prob_batch = self._actor(next_state_batch)

        next_state_action_values_1 = self._critic_1_target.get_q_values(
            next_state_batch,
            next_action_batch,
        ).view(
            (self.batch_size, 1)
        )  # (batch_size x 1)
        next_state_action_values_2 = self._critic_2_target.get_q_values(
            next_state_batch,
            next_action_batch,
        ).view(
            (self.batch_size, 1)
        )  # (batch_size x 1)
        next_state_action_values = torch.min(
            next_state_action_values_1, next_state_action_values_2
        )  # (batch_size x 1)

        # Entropy Regularization
        next_state_action_values = (
            next_state_action_values - self._entropy_coef * next_action_prob_batch
        )
        # (batch_size x 1)

        return next_state_action_values.view(-1)

    def _actor_learn_batch(self, batch: TransitionBatch) -> None:
        state_batch = batch.state  # (batch_size x state_dim)

        batch_size = state_batch.shape[0]
        # sanity check they have same batch_size
        assert state_batch.shape[0] == batch_size

        action_batch, action_prob_batch = self._actor(state_batch)  # (batch_size)

        state_action_values_1 = self._critic_1.get_q_values(
            state_batch, action_batch
        ).view(
            (self.batch_size, 1)
        )  # (batch_size)

        state_action_values_2 = self._critic_2.get_q_values(
            state_batch, action_batch
        ).view(
            (self.batch_size, 1)
        )  # (batch_size)

        state_action_values = torch.min(
            state_action_values_1, state_action_values_2
        )  # (batch_size)

        policy_loss = (
            self._entropy_coef * torch.log(action_prob_batch) - state_action_values
        ).mean()

        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()

    # overide parent class method 'act' which is for discrete case (output probs of actions)
    # In discrete SAC, actor takes state and outputs a float number action.
    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = True,
    ) -> Action:

        with torch.no_grad():
            subjective_state_tensor = (
                subjective_state
                if isinstance(subjective_state, torch.Tensor)
                else torch.tensor(subjective_state, dtype=torch.float32)
            )  # ([batch_size x ] state_dim)  # batch dimension only occurs if subjective_state is a batch

            subjective_state_tensor = subjective_state_tensor.to(self.device)

            action, _ = self._actor(subjective_state_tensor)  # ([batch_size x] 1)

        if exploit:
            return action
        else:
            # NOTE: Continuous SAC in nature sampled from a stochastic policy as exploration. Other exploration modules could be applied here, e.g. EGreedyExplore.
            return action

    # actor network.
    def make_specified_network(
        self,
        output_dim: int = 1,  # 1 dim action (e.g. Pendulum env)
    ):
        return self.network_type(
            input_dim=self.state_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            action_bound=self.action_space.high[0],  # 1 dim action
        )
