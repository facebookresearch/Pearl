import torch
import torch.nn.functional as F

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.state import SubjectiveState
from pearl.core.common.replay_buffer.replay_buffer import ReplayBuffer


class TensorBasedReplayBuffer(ReplayBuffer):
    @staticmethod
    def _process_single_state(state: SubjectiveState) -> torch.tensor:
        return torch.tensor(state).unsqueeze(0)  # (1 x state_dim)

    @staticmethod
    def _process_single_action(
        action: Action, action_space: ActionSpace
    ) -> torch.tensor:
        return F.one_hot(
            torch.tensor([action]), num_classes=action_space.n
        )  # (1 x action_dim)

    @staticmethod
    def _process_single_reward(reward: float) -> torch.tensor:
        return torch.tensor([reward])

    @staticmethod
    def _process_single_done(done: bool) -> torch.tensor:
        return torch.tensor([done]).float()  # (1)

    @staticmethod
    def _create_action_tensor_and_mask(
        action_space: ActionSpace, available_actions: ActionSpace
    ) -> (torch.tensor, torch.tensor):
        available_actions_tensor_with_padding = torch.zeros(
            (1, action_space.n, action_space.n)
        )  # (1 x action_space_size x action_dim)
        available_actions_tensor = F.one_hot(
            torch.arange(0, available_actions.n), num_classes=action_space.n
        )  # (1 x available_action_space_size x action_dim)
        available_actions_tensor_with_padding[
            0, : available_actions.n, :
        ] = available_actions_tensor
        available_actions_mask = torch.zeros(
            (1, action_space.n)
        )  # (1 x action_space_size)
        available_actions_mask[0, available_actions.n :] = 1
        available_actions_mask = available_actions_mask.bool()

        return (available_actions_tensor_with_padding, available_actions_mask)
