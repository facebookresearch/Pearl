from typing import Any, Dict, Iterable

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.neural_networks.common.utils import init_weights
from pearl.neural_networks.sequential_decision_making.actor_networks import (
    ActorNetworkType,
    VanillaActorNetwork,
)
from pearl.policy_learners.exploration_modules.common.propensity_exploration import (
    PropensityExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
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
        # pyre-fixme[9]: exploration_module has type `ExplorationModule`; used as
        #  `None`.
        exploration_module: ExplorationModule = None,
        learning_rate: float = 0.0001,
        discount_factor: float = 0.99,
        batch_size: int = 128,
        network_type: ActorNetworkType = VanillaActorNetwork,
        on_policy: bool = True,
        is_action_continuous: bool = False,
    ) -> None:
        super(PolicyGradient, self).__init__(
            # TODO: Replace to probability exploration module
            exploration_module=exploration_module
            if exploration_module is not None
            else PropensityExploration(),
            training_rounds=1,  # PG must set this to 1
            batch_size=batch_size,
            on_policy=on_policy,
            is_action_continuous=is_action_continuous,
        )
        self._action_space = action_space
        self._learning_rate = learning_rate
        self.network_type = network_type
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.action_space = action_space

        # pyre-fixme[4]: Attribute must be annotated.
        self._actor = self.make_specified_network()
        self._actor.apply(init_weights)
        self._actor_optimizer = optim.AdamW(
            self._actor.parameters(), lr=learning_rate, amsgrad=True
        )
        self._discount_factor = discount_factor

    # TODO: Assumes Gym interface, fix it.
    # pyre-fixme[3]: Return type must be annotated.
    def make_specified_network(
        self,
    ):
        # pyre-fixme[28]: Unexpected keyword argument `input_dim`.
        return self.network_type(
            input_dim=self.state_dim,
            hidden_dims=self.hidden_dims,
            # pyre-fixme[16]: `ActionSpace` has no attribute `n`.
            output_dim=self.action_space.n,
        )

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        # TODO: Assumes gym action space.
        # Fix the available action space.
        with torch.no_grad():
            subjective_state = subjective_state.view((1, -1))
            # (1 x state_dim)

            action_probabilities = self._actor(subjective_state)
            # (action_space_size, 1)

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
        return_batch = batch.cum_reward  # (batch_size)

        batch_size = state_batch.shape[0]
        # sanity check they have same batch_size
        assert action_batch.shape[0] == batch_size
        # pyre-fixme
        assert return_batch.shape[0] == batch_size

        policy_propensities = self._get_action_prob(batch.state, batch.action)
        negative_log_probs = -torch.log(policy_propensities + 1e-8)
        loss = torch.sum(
            negative_log_probs * batch.reward
        )  # reward in batch is cummulated return
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return {"loss": loss.mean().item()}

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _get_action_prob(self, state_batch, action_batch, actor=None):
        if actor is None:
            action_probs = self._actor(state_batch)
        else:
            action_probs = actor(state_batch)
        # TODO action_batch is one-hot encoding vectors
        return torch.sum(action_probs * action_batch, dim=1, keepdim=True)
