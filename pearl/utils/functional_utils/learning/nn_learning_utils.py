from typing import Dict

import torch
from pearl.neural_networks.sequential_decision_making.twin_critic import TwinCritic

# TODO: this function should be moved to actor_critic_base once all actor critic classes
# inherits from it.


def optimize_twin_critics_towards_target(
    twin_critic: TwinCritic,
    optimizer: torch.optim.Optimizer,
    state_batch: torch.Tensor,
    action_batch: torch.Tensor,
    expected_target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Performs an optimization step on the two critic networks towards a given target.
    Args:
        twin_critic: the twin critics to be optimized.
        optimizer: the optimizer used for training.
        state_batch: a batch of states with shape (batch_size, state_dim)
        action_batch: a batch of actions with shape (batch_size, action_dim)
        expected_target: the batch of target estimates.
    Returns:
        List[torch.Tensor]: individual critic losses along with the mean loss.
    """
    criterion = torch.nn.MSELoss()
    optimizer.zero_grad()
    q_1, q_2 = twin_critic.get_twin_critic_values(state_batch, action_batch)
    loss = criterion(q_1, expected_target) + criterion(q_2, expected_target)
    loss.backward()
    optimizer.step()

    return {
        "mean_loss": loss.item(),
        "critic_1_values": q_1.mean().item(),
        "critic_2_values": q_2.mean().item(),
    }
