from typing import Any, Dict, List, Optional, Type

import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.utils import update_target_network
from pearl.neural_networks.common.value_networks import VanillaQValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
    VanillaContinuousActorNetwork,
)
from pearl.neural_networks.sequential_decision_making.q_value_network import (
    QValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    make_critic,
    twin_critic_action_value_update,
    update_critic_target_network,
)
from pearl.policy_learners.sequential_decision_making.ddpg import (
    DeepDeterministicPolicyGradient,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
from torch import nn, optim


class TD3(DeepDeterministicPolicyGradient):
    """
    TD3 uses a deterministic actor, Twin critics, and a delayed actor update.
        - An exploration module is used with deterministic actors.
        - To avoid exploration, use NoExploration module.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: List[int],
        critic_hidden_dims: List[int],
        exploration_module: Optional[ExplorationModule] = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: Type[ActorNetwork] = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        actor_update_noise: float = 0.2,
        actor_update_noise_clip: float = 0.5,
        action_representation_module: Optional[ActionRepresentationModule] = None,
    ) -> None:
        assert isinstance(action_space, BoxActionSpace)
        super(TD3, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            exploration_module=exploration_module,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            action_representation_module=action_representation_module,
        )
        self._action_space: BoxActionSpace = action_space
        self._actor_update_freq = actor_update_freq
        self._actor_update_noise = actor_update_noise
        self._actor_update_noise_clip = actor_update_noise_clip
        self._critic_update_count = 0

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        self._critic_learn_batch(batch)  # critic update
        self._critic_update_count += 1

        # delayed actor update
        if self._critic_update_count % self._actor_update_freq == 0:
            # see ddpg base class for actor update details
            self._actor_learn_batch(batch)

            # update targets of critics using soft updates
            update_critic_target_network(
                self._critic_target,
                self._critic,
                self._use_twin_critic,
                self._critic_soft_update_tau,
            )
            # update target of actor network using soft updates
            update_target_network(
                self._actor_target, self._actor, self._actor_soft_update_tau
            )

        return {}

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        with torch.no_grad():
            # sample next_action from actor's target network; shape (batch_size, action_dim)
            next_action = self._actor_target.sample_action(batch.next_state)

            # sample clipped gaussian noise
            noise = torch.normal(
                mean=0,
                std=self._actor_update_noise,
                size=next_action.size(),
                device=batch.device,
            )

            noise = torch.clamp(
                noise,
                -self._actor_update_noise_clip,
                self._actor_update_noise_clip,
            )  # shape (batch_size, action_dim)

            # add clipped noise to next_action
            low = torch.tensor(self._action_space.low, device=batch.device)
            high = torch.tensor(self._action_space.high, device=batch.device)

            next_action = torch.clamp(
                next_action + noise, low, high
            )  # shape (batch_size, action_dim)

            # sample q values of (next_state, next_action) from targets of critics
            next_q1, next_q2 = self._critic_target.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network),
            #                  Qtarget_2(s', a from target actor network)})
            expected_state_action_values = (
                next_q * self._discount_factor * (1 - batch.done.float())
            ) + batch.reward  # (batch_size)

        # update twin critics towards bellman target
        assert isinstance(self._critic, TwinCritic)
        loss_critic_update = twin_critic_action_value_update(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            optimizer=self._critic_optimizer,
            critic=self._critic,
        )
        return loss_critic_update


class RCTD3(TD3):
    """
    RCTD3 uses TD3 based implementation for reward constraint optimization.
        - An exploration module is used with deterministic actors.
        - To avoid exploration, use NoExploration module.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: List[int],
        critic_hidden_dims: List[int],
        exploration_module: Optional[ExplorationModule] = None,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_network_type: Type[ActorNetwork] = VanillaContinuousActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
        actor_soft_update_tau: float = 0.005,
        critic_soft_update_tau: float = 0.005,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        actor_update_noise: float = 0.2,
        actor_update_noise_clip: float = 0.5,
        lambda_constraint: float = 1.0,
        cost_discount_factor: float = 0.5,
    ) -> None:
        super(RCTD3, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            exploration_module=exploration_module,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            actor_soft_update_tau=actor_soft_update_tau,
            critic_soft_update_tau=critic_soft_update_tau,
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            actor_update_freq=actor_update_freq,
            actor_update_noise=actor_update_noise,
            actor_update_noise_clip=actor_update_noise_clip,
        )
        self.lambda_constraint = lambda_constraint
        self.cost_discount_factor = cost_discount_factor

        # initialize cost critic
        self.cost_critic: nn.Module = make_critic(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_dims=critic_hidden_dims,
            use_twin_critic=self._use_twin_critic,
            network_type=critic_network_type,
        )
        self._cost_critic_optimizer = optim.AdamW(
            [
                {
                    "params": self.cost_critic.parameters(),
                    "lr": critic_learning_rate,
                    "amsgrad": True,
                },
            ]
        )
        self.target_of_cost_critic: nn.Module = make_critic(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_dims=critic_hidden_dims,
            use_twin_critic=self._use_twin_critic,
            network_type=critic_network_type,
        )
        update_critic_target_network(
            self.target_of_cost_critic,
            self.cost_critic,
            self._use_twin_critic,
            1,
        )

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        # update critics
        res = self._critic_learn_batch(batch)
        self._critic_update_count += 1

        if self._critic_update_count % 500 == 0:
            print(
                "constraint value: {} \n lambda: {} \n critic_mean_loss: {} \n cost_critic_mean_loss: {} \n critic_mean: {} cost_critic_mean: {}".format(
                    self.safety_module.constraint_value,
                    self.lambda_constraint,
                    res["critic_mean_loss"],
                    res["cost_critic_mean_loss"],
                    res["critic_1_values"],
                    res["cost_critic_2_values"],
                )
            )

        # update lambda to the current value of safety module
        self.lambda_constraint = self.safety_module.lambda_constraint

        # delayed actor update
        if self._critic_update_count % self._actor_update_freq == 0:
            # see ddpg base class for actor update details
            self._actor_learn_batch(batch)

            # update targets of twin critics using soft updates
            update_critic_target_network(
                self._critic_target,
                self._critic,
                self._use_twin_critic,
                self._critic_soft_update_tau,
            )

            # update targets of cost twin critics using soft updates
            update_critic_target_network(
                self.target_of_cost_critic,
                self.cost_critic,
                self._use_twin_critic,
                self._critic_soft_update_tau,
            )

            # update target of actor network using soft updates
            update_target_network(
                self._actor_target, self._actor, self._actor_soft_update_tau
            )

        return {}

    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:

        # sample a batch of actions from the actor network; shape (batch_size, action_dim)
        action_batch = self._actor.sample_action(batch.state)

        # samples q values for (batch.state, action_batch) from twin critics
        q1, q2 = self._critic.get_q_values(
            state_batch=batch.state, action_batch=action_batch
        )
        # clipped double q learning (reduce overestimation bias); shape (batch_size)
        q = torch.minimum(q1, q2)

        # samples cost q values for (batch.state, action_batch) from twin critics
        cost_q1, cost_q2 = self.cost_critic.get_q_values(
            state_batch=batch.state, action_batch=action_batch
        )
        # clipped double q learning (reduce overestimation bias); shape (batch_size)
        cost_q = torch.maximum(cost_q1, cost_q2)
        # optimization objective: optimize actor to maximize Q(s, a)
        loss = -(q.mean() - self.lambda_constraint * cost_q.mean())
        # print("actor loss q mean:{}".format(q.mean().item()))
        # print("actor loss cost q mean:{}".format(cost_q.mean().item()))

        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return {"actor_loss": loss.mean().item()}

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        res = {}
        train_critic_res = self._critic_custom_learn_batch(
            batch,
            critic=self._critic,
            target_of_critic=self._critic_target,
            critic_optimizer=self._critic_optimizer,
            discount_factor=self._discount_factor,
            critic_key="reward",
        )
        # TODO: take max instead of min in the cost critic?
        train_cost_critic_res = self._critic_custom_learn_batch(
            batch,
            critic=self.cost_critic,
            target_of_critic=self.target_of_cost_critic,
            critic_optimizer=self._cost_critic_optimizer,
            discount_factor=self.cost_discount_factor,
            critic_key="cost",
        )
        # modify the keys name
        train_cost_critic_res_new = {}
        for key, value in train_cost_critic_res.items():
            train_cost_critic_res_new["cost_{}".format(key)] = value
        res.update(train_critic_res)
        res.update(train_cost_critic_res_new)
        return res

    def _critic_custom_learn_batch(
        self,
        batch: TransitionBatch,
        critic: nn.Module,
        target_of_critic: nn.Module,
        critic_optimizer: optim.Optimizer,
        discount_factor: float,
        critic_key: str = "reward",
    ) -> Dict[str, Any]:

        assert critic_key in ["reward", "cost"]
        with torch.no_grad():
            # sample next_action from actor's target network; shape (batch_size, action_dim)
            next_action = self._actor_target.sample_action(batch.next_state)

            # sample clipped gaussian noise
            noise = torch.normal(
                mean=0,
                std=self._actor_update_noise,
                size=next_action.size(),
                device=batch.device,
            )

            noise = torch.clamp(
                noise,
                -self._actor_update_noise_clip,
                self._actor_update_noise_clip,
            )  # shape (batch_size, action_dim)

            # add clipped noise to next_action
            low, high = torch.tensor(
                self._action_space.low, device=batch.device
            ), torch.tensor(self._action_space.high, device=batch.device)

            next_action = torch.clamp(
                next_action + noise, low, high
            )  # shape (batch_size, action_dim)

            # sample q values of (next_state, next_action) from targets of twin critics
            next_q1, next_q2 = target_of_critic.get_q_values(
                state_batch=batch.next_state,
                action_batch=next_action,
            )  # shape (batch_size)

            # clipped double q learning (reduce overestimation bias)
            next_q = torch.minimum(next_q1, next_q2)

            # compute bellman target:
            # r + gamma * (min{Qtarget_1(s', a from target actor network), Qtarget_2(s', a from target actor network)}) no-qa
            reward_or_cost = batch.reward if critic_key == "reward" else batch.cost

            expected_state_action_values = (
                next_q * discount_factor * (1 - batch.done.float())
            ) + reward_or_cost  # (batch_size)

        # update critics towards bellman target

        loss_critic_update = twin_critic_action_value_update(
            state_batch=batch.state,
            action_batch=batch.action,
            expected_target_batch=expected_state_action_values,
            optimizer=critic_optimizer,
            # pyre-fixme
            critic=critic,
        )
        return loss_critic_update
