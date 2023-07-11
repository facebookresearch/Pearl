import copy
from typing import Any, Dict, Iterable

import torch

from pearl.api.action_space import ActionSpace
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
        self._actor_old = copy.deepcopy(self._actor)

    def _define_loss(self, batch: TransitionBatch) -> torch.Tensor:
        action_probs = self._get_action_prob(batch.state, batch.action)
        with torch.no_grad():
            action_probs_old = self._get_action_prob(
                batch.state, batch.action, self._actor_old
            )
        r_thelta = torch.div(action_probs, action_probs_old)

        clip = torch.clamp(r_thelta, min=1.0 - self._epsilon, max=1.0 + self._epsilon)

        with torch.no_grad():
            advantage = self._define_advantage(batch)

        return torch.sum(-torch.min(r_thelta * advantage, clip * advantage))

    def _define_advantage(self, batch: TransitionBatch) -> torch.Tensor:
        # TODO add in next diff
        return batch.reward

    def learn(self, replay_buffer: ReplayBuffer) -> Dict[str, Any]:
        super(ProximalPolicyOptimization, self).learn(replay_buffer)
        # update old actor with latest actor for next round
        self._actor_old.load_state_dict(self._actor.state_dict())
