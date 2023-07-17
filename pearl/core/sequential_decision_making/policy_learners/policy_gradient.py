from typing import Any, Dict, Iterable

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.core.common.neural_networks.utils import init_weights
from pearl.core.common.policy_learners.exploration_module.exploration_module import (
    ExplorationModule,
)
from pearl.core.common.policy_learners.policy_learner import PolicyLearner
from pearl.core.common.replay_buffer.transition import TransitionBatch
from pearl.core.sequential_decision_making.neural_networks.actor_networks import (
    ActorNetworkType,
    VanillaActorNetwork,
)
from pearl.core.sequential_decision_making.policy_learners.exploration_module.propensity_exploration import (
    PropensityExploration,
)
from torch import optim


# TODO: Only support discrete action space problems for now and assumes Gym action space.
# Currently available actions is not used. Needs to be updated once we know the input structure
# of production stack on this param.
class PolicyGradient(PolicyLearner):
    """
    An Abstract Class for Deep Policy Gradient policy learner.
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
        network_type: ActorNetworkType = VanillaActorNetwork,
        on_policy: bool = True,
    ) -> None:
        super(PolicyGradient, self).__init__(
            # TODO: Replace to probability exploration module
            exploration_module=exploration_module
            if exploration_module is not None
            else PropensityExploration(),
            training_rounds=training_rounds,
            batch_size=batch_size,
            on_policy=on_policy,
        )
        self._action_space = action_space
        self._learning_rate = learning_rate

        # TODO: Assumes Gym interface, fix it.
        def make_specified_network():
            return network_type(
                input_dim=state_dim,
                hidden_dims=hidden_dims,
                output_dim=action_space.n,
            )

        self._actor = make_specified_network()
        self._actor.apply(init_weights)
        self._actor_optimizer = optim.AdamW(
            self._actor.parameters(), lr=learning_rate, amsgrad=True
        )
        self._discount_factor = discount_factor

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        # TODO: Assumes subjective state is a torch tensor and gym action space.
        # Fix the available action space.
        with torch.no_grad():
            subjective_state_tensor = torch.tensor(subjective_state).view(
                (1, -1)
            )  # (1 x state_dim)
            action_probabilities = self._actor(
                subjective_state_tensor
            )  # (action_space_size, 1)
            exploit_action = torch.argmax(action_probabilities).view((-1)).item()

        if exploit:
            return exploit_action

        return self._exploration_module.act(
            subjective_state,
            available_action_space,
            values=action_probabilities,
            exploit_action=exploit_action,
        )

    # TODO: Currently does not support discount factor. Needs to add to replay buffer
    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        state_batch = batch.state  # (batch_size x state_dim)
        action_batch = batch.action  # (batch_size x action_dim)
        return_batch = batch.reward  # (batch_size)

        batch_size = state_batch.shape[0]
        # sanity check they have same batch_size
        assert action_batch.shape[0] == batch_size
        assert return_batch.shape[0] == batch_size

        loss = self._define_loss(batch)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return {"loss": loss.mean().item()}

    def _get_action_prob(self, state_batch, action_batch, actor=None):
        if actor is None:
            action_probs = self._actor(state_batch)
        else:
            action_probs = actor(state_batch)
        # TODO action_batch is one-hot encoding vectors
        return torch.sum(action_probs * action_batch, dim=1, keepdim=True)

    def _define_loss(self, batch: TransitionBatch) -> torch.Tensor:
        policy_propensities = self._get_action_prob(batch.state, batch.action)

        negative_log_probs = -torch.log(policy_propensities)
        return torch.sum(
            negative_log_probs * batch.reward
        )  # reward in batch is cummulated return
