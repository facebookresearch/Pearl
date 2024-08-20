import torch
from typing import Optional

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.policy_learners.exploration_modules.common.exploration_base import ExplorationBase
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace


class EntropyRegularizationExploration(ExplorationBase):
    """
    Entropy Regularization exploration module.
    Encourages exploration by incorporating entropy into action selection.
    """

    def __init__(self, entropy_coeff: float) -> None:
        super(EntropyRegularizationExploration, self).__init__()
        self.entropy_coeff = entropy_coeff

    def calculate_entropy(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the entropy of the action probability distribution.
        """
        return -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)

    def act(
            self,
            subjective_state: SubjectiveState,
            action_space: ActionSpace,
            action_probs: Optional[torch.Tensor],
            values: Optional[torch.Tensor] = None,
            action_availability_mask: Optional[torch.Tensor] = None,
            representation: Optional[torch.nn.Module] = None,
    ) -> Action:
        if action_probs is None:
            raise ValueError("action_probs cannot be None for entropy-based exploration")
        if not isinstance(action_space, DiscreteActionSpace):
            raise TypeError("action space must be discrete")

        # Calculate entropy
        entropy = self.calculate_entropy(action_probs)

        # Adjust action probabilities using entropy coefficient
        adjusted_probs = action_probs + self.entropy_coeff * entropy
        adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)  # Normalize

        # Sample an action based on the adjusted probabilities
        action_idx = torch.multinomial(adjusted_probs, 1).item()
        return action_space.actions[action_idx].to(action_probs.device)
