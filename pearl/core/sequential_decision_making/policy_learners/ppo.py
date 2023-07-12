import copy
from typing import Any, Dict, Iterable

import torch

from pearl.api.action_space import ActionSpace
from pearl.core.common.neural_networks.value_networks import VanillaValueNetwork
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.core.common.replay_buffer.replay_buffer import ReplayBuffer
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.core.sequential_decision_making.neural_networks.actor_networks import (
    ActorNetworkType,
    VanillaActorNetwork,
)
from pearl.core.sequential_decision_making.policy_learners.policy_gradient import (
    PolicyGradient,
)
from torch import optim


class ProximalPolicyOptimization(PolicyGradient):
    """
    paper: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_dims: Iterable[int],
        exploration_module: ExplorationModule = None,
        learning_rate: float = 0.0001,
        training_rounds: int = 100,
        batch_size: int = 128,
        network_type: ActorNetworkType = VanillaActorNetwork,
        epsilon: float = 0.0,
        critic_loss_scaling: float = 0.5,
        entropy_bonus_scaling: float = 0.01,
    ) -> None:
        super(ProximalPolicyOptimization, self).__init__(
            exploration_module=exploration_module,
            training_rounds=training_rounds,
            batch_size=batch_size,
            action_space=action_space,
            learning_rate=learning_rate,
            network_type=network_type,
            state_dim=state_dim,
            hidden_dims=hidden_dims,
        )
        self._epsilon = epsilon
        self._critic_loss_scaling = critic_loss_scaling
        self._entropy_bonus_scaling = entropy_bonus_scaling
        self._actor_old = copy.deepcopy(self._actor)
        # V(s)
        def make_critic_network():
            return VanillaValueNetwork(
                input_dim=state_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )

        self._critic = make_critic_network()
        self._optimizer = optim.AdamW(
            [
                {
                    "params": self._actor.parameters(),
                    "lr": learning_rate,
                    "amsgrad": True,
                },
                {
                    "params": self._critic.parameters(),
                    "lr": learning_rate,
                    "amsgrad": True,
                },
            ]
        )

    def _define_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """
        Loss = actor loss + critic_loss_scaling * critic loss + entropy_bonus_scaling * entropy loss
        """
        vs = self._critic(batch.state)
        action_probs = self._get_action_prob(batch.state, batch.action)
        # actor loss
        with torch.no_grad():
            action_probs_old = self._get_action_prob(
                batch.state, batch.action, self._actor_old
            )
        r_thelta = torch.div(action_probs, action_probs_old)

        clip = torch.clamp(r_thelta, min=1.0 - self._epsilon, max=1.0 + self._epsilon)

        # advantage estimator, in paper:
        # A = sum(lambda^t*gamma^t*TD_error), while TD_error = reward + gamma * V(s+1) - V(s)
        # when lambda = 1 and gamma = 1
        # A = sum(TD_error) = return - V(s)
        # TODO support lambda and gamma
        with torch.no_grad():
            advantage = batch.reward - vs

        # critic loss
        criterion = torch.nn.MSELoss()
        vs_loss = criterion(vs, batch.reward)

        # entropy
        # Categorical is good for Cartpole Env where actions are discrete
        # TODO need to support continuous action
        entropy = torch.distributions.Categorical(action_probs.detach()).entropy()
        return (
            torch.sum(-torch.min(r_thelta * advantage, clip * advantage))
            + self._critic_loss_scaling * vs_loss
            - torch.sum(self._entropy_bonus_scaling * entropy)
        )

    def learn(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        super().learn(replay_buffer)
        # update old actor with latest actor for next round
        self._actor_old.load_state_dict(self._actor.state_dict())
