# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


"""Shared PPO losses, advantage math and native replay hooks.

The shared-base organization follows the maintainer's request in PEARL PR103
and acknowledges kuds' earlier continuous-PPO contribution. Native discrete
defaults/reductions/constructor semantics are retained; continuous/RPO provide
their explicit Gaussian behavior metadata and shuffled-rollout update hooks.
"""

import torch
from dataclasses import dataclass
from typing import Any, List
from pearl.neural_networks.common.value_networks import ValueNetwork
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.policy_learners.sequential_decision_making.actor_critic_base import (
    ActorCriticBase,
)
from pearl.replay_buffers.replay_buffer import ReplayBuffer
from pearl.replay_buffers.tensor_based_replay_buffer import TensorBasedReplayBuffer
from pearl.replay_buffers.transition import Transition, TransitionBatch
from pearl.utils.replay_buffer_utils import (
    make_replay_buffer_class_for_specific_transition_types,
)


@dataclass(frozen=False)
class PPOTransition(Transition):
    gae: torch.Tensor | None = None  # generalized advantage estimation
    lam_return: torch.Tensor | None = None  # lambda return
    action_probs: torch.Tensor | None = None  # action probs


@dataclass(frozen=False)
class PPOTransitionBatch(TransitionBatch):
    gae: torch.Tensor | None = None  # generalized advantage estimation
    lam_return: torch.Tensor | None = None  # lambda return
    action_probs: torch.Tensor | None = None  # action probs

    @classmethod
    def from_parent(
        cls,
        parent_obj: TransitionBatch,
        gae: torch.Tensor | None = None,
        lam_return: torch.Tensor | None = None,
        action_probs: torch.Tensor | None = None,
    ) -> "PPOTransitionBatch":
        # Extract attributes from parent_obj using __dict__ and create a new child object
        child_obj = cls(
            **parent_obj.__dict__,
            gae=gae,
            lam_return=lam_return,
            action_probs=action_probs,
        )
        return child_obj


PPOReplayBuffer: type[TensorBasedReplayBuffer] = (
    make_replay_buffer_class_for_specific_transition_types(
        PPOTransition, PPOTransitionBatch
    )
)


def ppo_advantage_step(
    reward,
    value,
    next_value,
    terminated,
    truncated,
    next_advantage,
    discount_factor,
    trace_decay,
):
    """One shared TD/trace step; only termination suppresses the value bootstrap."""
    delta = reward + discount_factor * next_value * (~terminated) - value
    return delta + discount_factor * trace_decay * next_advantage * (
        ~(terminated | truncated)
    )


@torch.no_grad()
def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    discount_factor: float = 0.99,
    trace_decay: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over [time, environments], with final-state truncation bootstrapping.

    next_values is evaluated at each transition's actual final observation,
    before autoreset. Trace propagation stops at both termination and truncation;
    only true termination suppresses the one-step value bootstrap.
    """
    if rewards.ndim != 2 or any(
        t.shape != rewards.shape for t in [values, next_values, terminated, truncated]
    ):
        raise ValueError("GAE inputs must share nonempty [time, environments] shape")
    if not rewards.numel() or not all(
        torch.isfinite(t).all() for t in [rewards, values, next_values]
    ):
        raise ValueError("GAE requires nonempty finite inputs")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("termination flags must be boolean")
    if not 0 <= discount_factor <= 1 or not 0 <= trace_decay <= 1:
        raise ValueError("discount and trace decay must lie in [0, 1]")
    result = torch.zeros_like(values)
    carry = torch.zeros_like(values[0])
    for time in reversed(range(rewards.shape[0])):
        carry = ppo_advantage_step(
            rewards[time],
            values[time],
            next_values[time],
            terminated[time],
            truncated[time],
            carry,
            discount_factor,
            trace_decay,
        )
        result[time] = carry
    return result, result + values


class ProximalPolicyOptimizationBase(ActorCriticBase):
    """Common executed math and replay hooks for discrete/continuous PPO."""

    @staticmethod
    def clipped_surrogate_loss(
        ratio, advantage, epsilon, reduction="mean", clipped_ratio=None
    ):
        if clipped_ratio is None:
            clipped_ratio = ratio.clamp(1 - epsilon, 1 + epsilon)
        terms = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
        if reduction == "mean":
            return terms.mean()
        if reduction == "sum":
            return terms.sum()
        raise ValueError("surrogate reduction must be mean or sum")

    @staticmethod
    def value_regression_loss(
        prediction, target, old_value=None, epsilon=None, half=True
    ):
        if old_value is None:
            result = torch.nn.functional.mse_loss(prediction, target)
        else:
            if epsilon is None:
                raise ValueError("clipped value loss requires epsilon")
            clipped = old_value + (prediction - old_value).clamp(-epsilon, epsilon)
            result = torch.maximum(
                (prediction - target).square(), (clipped - target).square()
            ).mean()
        return 0.5 * result if half else result

    def _critic_loss(self, batch: TransitionBatch) -> torch.Tensor:
        assert isinstance(batch, PPOTransitionBatch)
        assert batch.lam_return is not None
        if not isinstance(self._critic, ValueNetwork):
            raise TypeError("PPO state-value critic must implement ValueNetwork")
        prediction = self._critic(batch.state).reshape_as(batch.lam_return)
        return self.value_regression_loss(
            prediction, batch.lam_return.detach(), half=False
        )

    def learn(self, replay_buffer: ReplayBuffer) -> dict[str, Any]:
        self.preprocess_replay_buffer(replay_buffer)
        # sample from replay buffer and learn
        result = super().learn(replay_buffer)
        # update old actor with latest actor for next round
        return result

    def preprocess_replay_buffer(self, replay_buffer: ReplayBuffer) -> None:
        """
        Preprocess the replay buffer by calculating
        and adding the generalized advantage estimates (gae),
        truncated lambda returns (lam_return) and action probabilities (action_probs)
        under the current policy.
        See https://arxiv.org/abs/1707.06347 equation (11) for the definition of gae.
        See "Reinforcement Learning: An Introduction" by Sutton and Barto (2018) equation (12.10)
        for the definition of truncated lambda return.
        """
        assert isinstance(replay_buffer, TensorBasedReplayBuffer)
        assert len(replay_buffer.memory) > 0
        (
            state_list,
            action_list,
            available_actions_list,
            unavailable_actions_mask_list,
        ) = ([], [], [], [])
        for transition in reversed(replay_buffer.memory):
            state_list.append(transition.state)
            action_list.append(transition.action)
            available_actions_list.append(transition.curr_available_actions)
            unavailable_actions_mask_list.append(
                transition.curr_unavailable_actions_mask
            )
        history_summary_batch = self._history_summarization_module(
            torch.cat(state_list)
        ).detach()
        action_representation_batch = self.action_representation_module(
            torch.cat(action_list)
        )

        # Transitions in the reply buffer memory are in the CPU
        # (only sampled batches are moved to the used device, kept in replay_buffer.device)
        # To use it in expressions involving the models,
        # we must move them to the device being used first.
        history_summary_batch = history_summary_batch.to(
            replay_buffer.device_for_batches
        )
        action_representation_batch = action_representation_batch.to(
            replay_buffer.device_for_batches
        )

        state_values = self._critic(history_summary_batch).detach()
        action_probs = (
            # pyre-fixme[29]: `Union[Module, Tensor]` is not a function.
            self._actor.get_action_prob(
                state_batch=history_summary_batch,
                action_batch=action_representation_batch,
            )
            .detach()
            .unsqueeze(-1)
        )

        # Transitions in the reply buffer memory are in the CPU
        # (only sampled batches are moved to the used device,
        # kept in replay_buffer.device_for_batches)
        # To use it in expressions involving the critic,
        # we must move them to the device being used first.
        next_state = replay_buffer.memory[-1].next_state
        assert next_state is not None
        next_state_in_device = next_state.to(replay_buffer.device_for_batches)

        # Obtain the value of the most recent state stored in the replay buffer.
        # This value is used to compute the generalized advantage estimation (gae)
        # and the truncated lambda return for all states in the replay buffer.
        next_value = self._critic(
            self._history_summarization_module(next_state_in_device)
        ).detach()[
            0
        ]  # shape (1,)
        gae = torch.tensor([0.0]).to(state_values.device)
        for i, transition in enumerate(reversed(replay_buffer.memory)):
            original_transition_device = transition.device
            transition.to(state_values.device)
            gae = ppo_advantage_step(
                transition.reward,
                state_values[i],
                next_value,
                transition.terminated,
                transition.truncated,
                gae,
                self._discount_factor,
                self._trace_decay_param,
            )
            assert isinstance(transition, PPOTransition)
            transition.gae = gae
            # truncated lambda return of the state
            transition.lam_return = gae + state_values[i]
            # action probabilities from the current policy
            transition.action_probs = action_probs[i]
            next_value = state_values[i]
            transition.to(original_transition_device)

    def compare(self, other: PolicyLearner) -> str:
        """
        Compares two ProximalPolicyOptimization instances for equality.

        Args:
          other: The other PolicyLearner to compare with.

        Returns:
          str: A string describing the differences, or an empty string if they are identical.
        """

        differences: List[str] = []

        differences.append(super().compare(other))

        if not isinstance(other, ProximalPolicyOptimizationBase):
            differences.append("other is not an instance of ProximalPolicyOptimization")
        else:
            # Compare attributes specific to ProximalPolicyOptimization
            if self._epsilon != other._epsilon:
                differences.append(
                    f"_epsilon is different: {self._epsilon} vs {other._epsilon}"
                )
            if self._trace_decay_param != other._trace_decay_param:
                differences.append(
                    f"_trace_decay_param is different: {self._trace_decay_param} "
                    + f"vs {other._trace_decay_param}"
                )
            if self._entropy_bonus_scaling != other._entropy_bonus_scaling:
                differences.append(
                    f"_entropy_bonus_scaling is different: {self._entropy_bonus_scaling} "
                    + f"vs {other._entropy_bonus_scaling}"
                )

        return "\n".join(differences)
