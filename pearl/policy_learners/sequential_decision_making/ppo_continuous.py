import torch
from typing import Any, Dict, List, Optional, Type, Union
from torch.distributions import Normal

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.action_space import ActionSpace
from pearl.neural_networks.common.value_networks import (
    ValueNetwork,
    VanillaValueNetwork,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetwork,
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
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.sequential_decision_making.on_policy_replay_buffer import (
    OnPolicyReplayBuffer,
    OnPolicyTransition,
    OnPolicyTransitionBatch,
)
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.functional_utils.learning.critic_utils import (
    single_critic_state_value_loss,
)
from torch import nn


class ProximalPolicyOptimization(ActorCriticBase):
    """
    PPO algorithm for continuous action spaces.
    """

    def __init__(
            self,
            state_dim: int,
            action_space: ActionSpace,
            use_critic: bool,
            actor_hidden_dims: Optional[List[int]] = None,
            critic_hidden_dims: Optional[List[int]] = None,
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 1e-4,
            exploration_module: Optional[ExplorationModule] = None,
            actor_network_type: Type[ActorNetwork] = VanillaActorNetwork,
            critic_network_type: Type[ValueNetwork] = VanillaValueNetwork,
            discount_factor: float = 0.99,
            training_rounds: int = 100,
            batch_size: int = 128,
            epsilon: float = 0.0,
            trace_decay_param: float = 0.95,
            entropy_bonus_scaling: float = 0.01,
            action_representation_module: Optional[ActionRepresentationModule] = None,
            actor_network_instance: Optional[ActorNetwork] = None,
            critic_network_instance: Optional[Union[ValueNetwork, nn.Module]] = None,
    ) -> None:
        super(ProximalPolicyOptimization, self).__init__(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=actor_hidden_dims,
            use_critic=use_critic,
            critic_hidden_dims=critic_hidden_dims,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_actor_target=False,
            use_critic_target=False,
            actor_soft_update_tau=0.0,  # not used
            critic_soft_update_tau=0.0,  # not used
            use_twin_critic=False,
            exploration_module=(
                exploration_module
                if exploration_module is not None
                else PropensityExploration()
            ),
            discount_factor=discount_factor,
            training_rounds=training_rounds,
            batch_size=batch_size,
            is_action_continuous=True,  # Change to continuous action space
            on_policy=True,
            action_representation_module=action_representation_module,
            actor_network_instance=actor_network_instance,
            critic_network_instance=critic_network_instance,
        )
        self._epsilon = epsilon
        self._trace_decay_param = trace_decay_param
        self._entropy_bonus_scaling = entropy_bonus_scaling

    def _actor_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Loss = actor loss + critic loss + entropy_bonus_scaling * entropy loss
        """
        assert isinstance(batch, OnPolicyTransitionBatch)

        # Get action mean and standard deviation (for Gaussian distribution)
        action_mean, action_log_std = self._actor(batch.state)
        action_std = action_log_std.exp()

        # Create a normal distribution based on the mean and std
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(batch.action).sum(axis=-1)

        # Calculate the ratio for PPO
        action_probs_old = batch.action_probs
        assert action_probs_old is not None
        r_theta = torch.exp(log_probs - action_probs_old)  # shape (batch_size)

        # Clipped surrogate objective
        clip = torch.clamp(
            r_theta, min=1.0 - self._epsilon, max=1.0 + self._epsilon
        )
        loss = -torch.min(r_theta * batch.gae, clip * batch.gae).mean()

        # Entropy for encouraging exploration
        entropy = dist.entropy().sum(axis=-1).mean()
        loss -= self._entropy_bonus_scaling * entropy

        return loss

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        assert isinstance(batch, OnPolicyTransitionBatch)
        assert batch.lam_return is not None
        return single_critic_state_value_loss(
            state_batch=batch.state,
            expected_target_batch=batch.lam_return,
            critic=self._critic,
        )

    def learn(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        self.preprocess_replay_buffer(replay_buffer)
        # Sample from replay buffer and learn
        result = super().learn(replay_buffer)
        # Update old actor with the latest actor for the next round
        return result

    def preprocess_replay_buffer(self, replay_buffer: ReplayBuffer) -> None:
        """
        Preprocess the replay buffer by calculating
        and adding the generalized advantage estimates (gae),
        truncated lambda returns (lam_return), and action log probabilities (action_probs)
        under the current policy.
        """
        assert type(replay_buffer) is OnPolicyReplayBuffer
        assert len(replay_buffer.memory) > 0

        state_list, action_list = [], []
        for transition in reversed(replay_buffer.memory):
            state_list.append(transition.state)
            action_list.append(transition.action)

        history_summary_batch = self._history_summarization_module(
            torch.cat(state_list)
        ).detach()
        action_representation_batch = self._action_representation_module(
            torch.cat(action_list)
        )

        history_summary_batch = history_summary_batch.to(
            replay_buffer.device_for_batches
        )
        action_representation_batch = action_representation_batch.to(
            replay_buffer.device_for_batches
        )

        state_values = self._critic(history_summary_batch).detach()

        # Get action mean and standard deviation for current policy
        action_mean, action_log_std = self._actor(history_summary_batch)
        action_std = action_log_std.exp()
        dist = Normal(action_mean, action_std)

        # Calculate action log probabilities
        action_log_probs = dist.log_prob(action_representation_batch).sum(axis=-1).detach().unsqueeze(-1)

        next_state = replay_buffer.memory[-1].next_state
        assert next_state is not None
        next_state_in_device = next_state.to(replay_buffer.device_for_batches)
        next_value = self._critic(
            self._history_summarization_module(next_state_in_device)
        ).detach()[0]

        gae = torch.tensor([0.0]).to(state_values.device)
        for i, transition in enumerate(reversed(replay_buffer.memory)):
            original_transition_device = transition.device
            transition.to(state_values.device)
            td_error = (
                    transition.reward
                    + self._discount_factor * next_value * (~transition.terminated)
                    - state_values[i]
            )
            gae = (
                    td_error
                    + self._discount_factor
                    * self._trace_decay_param
                    * (~transition.terminated)
                    * gae
            )
            assert isinstance(transition, OnPolicyTransition)
            transition.gae = gae
            transition.lam_return = gae + state_values[i]
            transition.action_probs = action_log_probs[i]
            next_value = state_values[i]
            transition.to(original_transition_device)
