from typing import Any, Dict, Iterable, Optional, Type

from pearl.neural_networks.common.value_networks import ValueNetwork

from pearl.neural_networks.sequential_decision_making.actor_networks import ActorNetwork

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym  # noqa
import torch

from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.value_networks import VanillaValueNetwork
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    VanillaActorNetwork,
)
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
    single_critic_state_value_update,
)
from pearl.replay_buffers.transition import TransitionBatch


class REINFORCE(ActorCriticBase):
    """
    Williams, R. J. (1992). Simple statistical gradient-following algorithms
    for connectionist reinforcement learning. Machine learning, 8, 229-256.
    The critic serves as the baseline.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        actor_hidden_dims: Iterable[int],
        critic_hidden_dims: Optional[Iterable[int]],
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        actor_network_type: Type[ActorNetwork] = VanillaActorNetwork,
        critic_network_type: Type[ValueNetwork] = VanillaValueNetwork,
        exploration_module: Optional[ExplorationModule] = None,
        discount_factor: float = 0.99,
        training_rounds: int = 1,
    ) -> None:
        super(REINFORCE, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            # pyre-fixme: super class expects a QValueNetwork here,
            # but this class apparently requires a ValueNetwork
            # (replacing the type and default value to QValueNetworks break tests)
            critic_network_type=critic_network_type,
            use_actor_target=False,
            use_critic_target=False,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=0.0,  # not used
            use_twin_critic=False,
            exploration_module=exploration_module
            if exploration_module is not None
            else PropensityExploration(),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=0,  # REINFORCE does not use batch size
            is_action_continuous=False,
            on_policy=True,
        )

    def _actor_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        state_batch = (
            batch.state
        )  # (batch_size x state_dim) note that here batch_size = episode length
        return_batch = batch.cum_reward  # (batch_size)
        policy_propensities = self._actor.get_action_prob(
            batch.state, batch.action
        )  # shape (batch_size)
        negative_log_probs = -torch.log(policy_propensities + 1e-8)
        if self._use_critic:
            v = self._critic(state_batch).view(-1)  # (batch_size)
            assert return_batch is not None
            loss = torch.sum(negative_log_probs * (return_batch - v.detach()))
        else:
            loss = torch.sum(negative_log_probs * return_batch)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return {"actor_loss": loss.mean().item()}

    def _critic_learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        if self._use_critic:
            assert batch.cum_reward is not None
            return single_critic_state_value_update(
                state_batch=batch.state,
                expected_target_batch=batch.cum_reward,
                optimizer=self._critic_optimizer,
                critic=self._critic,
            )
        return {}
