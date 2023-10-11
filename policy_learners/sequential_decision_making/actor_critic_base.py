# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from abc import abstractmethod
from typing import Any, Dict, Iterable, Type

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

import torch

from pearl.api.action import Action

from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
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
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.replay_buffers.transition import TransitionBatch
from pearl.utils.instantiations.action_spaces.action_spaces import DiscreteActionSpace
from torch import optim


class OffPolicyActorCritic(PolicyLearner):
    """
    A base class for all actor-critic based policy learners. Many components are common to actor-critic methods.
        - Actor and critic (as well as target networks) network initializations.
        - reset and learn_batch methods.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_dims: Iterable[int],
        critic_learning_rate: float = 1e-4,
        actor_learning_rate: float = 1e-4,
        # pyre-fixme[9]: exploration_module has type `ExplorationModule`; used as
        #  `None`.
        exploration_module: ExplorationModule = None,
        discount_factor: float = 0.99,
        training_rounds: int = 100,
        batch_size: int = 128,
        critic_soft_update_tau: float = 0.05,
        is_action_continuous: bool = False,  # can accommodate continuous action spaces but set to False by default
        actor_network_type: ActorNetworkType = VanillaActorNetwork,
        critic_network_type: Type[QValueNetwork] = VanillaQValueNetwork,
    ) -> None:
        super(OffPolicyActorCritic, self).__init__(
            on_policy=False,  # enforced to be an off policy policy learner
            is_action_continuous=is_action_continuous,
            action_space=action_space,
            training_rounds=training_rounds,
            batch_size=batch_size,
        )

        self._state_dim = state_dim
        if (
            type(action_space) == gym.spaces.discrete.Discrete
            or type(action_space) == DiscreteActionSpace
        ):
            # pyre-fixme[16]: `ActionSpace` has no attribute `shape`.
            self._action_dim: int = action_space.n
        elif type(action_space) == gym.spaces.box.Box:
            self._action_dim = action_space.shape[0]
        else:
            raise NotImplementedError("Action space not implemented")
        self.is_action_continuous = is_action_continuous
        self._exploration_module = exploration_module

        # pyre-fixme[3]: Return type must be annotated.
        def make_specified_actor_network():
            # pyre-fixme[28]: Unexpected keyword argument `input_dim`.
            return actor_network_type(
                input_dim=state_dim,
                hidden_dims=hidden_dims,
                output_dim=self._action_dim,
                action_space=action_space,
            )

        # actor network takes state as input and outputs an action vector
        # pyre-fixme[4]: Attribute must be annotated.
        self._actor = make_specified_actor_network()
        self._actor.apply(init_weights)
        self._actor_optimizer = optim.AdamW(
            self._actor.parameters(), lr=actor_learning_rate, amsgrad=True
        )

        # twin critic: using two separate critic networks to reduce overestimation bias
        # optimizers of two critics are alredy initialized in TwinCritic
        self._twin_critics = TwinCritic(
            state_dim=state_dim,
            action_dim=self._action_dim,
            hidden_dims=hidden_dims,
            network_type=critic_network_type,
            init_fn=init_weights,
        )

        # target networks of twin critics
        self._targets_of_twin_critics = TwinCritic(
            state_dim=state_dim,
            action_dim=self._action_dim,
            hidden_dims=hidden_dims,
            network_type=critic_network_type,
            init_fn=init_weights,
        )

        self._critic_optimizer = optim.AdamW(
            self._twin_critics.parameters(),
            lr=critic_learning_rate,
            amsgrad=True,
        )

        # target networks are initialized to parameters of the source network (tau is set to 1)
        update_target_networks(
            self._targets_of_twin_critics._critic_networks_combined,
            self._twin_critics._critic_networks_combined,
            tau=1,
        )

        self._critic_soft_update_tau = critic_soft_update_tau
        self._discount_factor = discount_factor

    """
    Act method is included here but since it is intrinsically tied to the type of exploration method
    as well as the type of action space (discrete and continuous), compatibility checks are required for implementation.
    """

    def act(
        self,
        subjective_state: SubjectiveState,
        available_action_space: ActionSpace,
        exploit: bool = False,
    ) -> Action:
        # TODO: Assumes subjective state is a torch tensor and gym action space.
        # Fix the available action space.

        # Step 1: compute exploit_action (action computed by actor network; and without any exploration)
        with torch.no_grad():
            subjective_state_tensor = (
                subjective_state
                if isinstance(subjective_state, torch.Tensor)
                else torch.tensor(subjective_state)
            )  # ([batch_size x ] state_dim)  # batch dimension only occurs if subjective_state is a batch

            subjective_state_tensor = subjective_state_tensor.to(self.device)

            if self.is_action_continuous:
                exploit_action, _ = self._actor.sample_action_and_get_log_prob(
                    subjective_state_tensor
                )
                action_probabilities = None
            else:
                action_probabilities = self._actor(
                    subjective_state_tensor
                )  # (action_space_size, 1)
                exploit_action = torch.argmax(action_probabilities).view((-1)).item()

        # Step 2: return exploit action if no exploration, else pass through the exploration module
        if exploit:
            return exploit_action

        return self._exploration_module.act(
            subjective_state,
            available_action_space,
            values=action_probabilities,
            exploit_action=exploit_action,
        )

    def reset(self, action_space: ActionSpace) -> None:
        self._action_space = action_space

    def learn_batch(self, batch: TransitionBatch) -> Dict[str, Any]:
        self._critic_learn_batch(batch)  # update critic
        self._actor_learn_batch(batch)  # update actor

        self._rounds += 1

        # update targets of twin critics using soft updates
        update_target_networks(
            self._targets_of_twin_critics._critic_networks_combined,
            self._twin_critics._critic_networks_combined,
            self._critic_soft_update_tau,
        )
        return {}

    @abstractmethod
    # pyre-fixme[2]: Parameter must be annotated.
    def _actor_learn_batch(self, batch: TransitionBatch, **options) -> Dict[str, Any]:
        pass

    @abstractmethod
    # pyre-fixme[2]: Parameter must be annotated.
    def _critic_learn_batch(self, batch: TransitionBatch, **options) -> Dict[str, Any]:
        pass
