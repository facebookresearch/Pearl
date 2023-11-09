import random
from typing import List

import torch

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace


class DiscreteActionSpace(ActionSpace):
    # pyre-fixme[3]: Return type must be annotated.
    def __init__(self, actions: List[Action]):
        """
        actions: List[Action] could be a list of action vector
        or a range of action index
        TODO better idea to write this cleaner?
        """
        self.actions = actions
        # pyre-fixme[4]: Attribute must be annotated.
        self.n = len(actions)

    def sample(self) -> Action:
        return random.choice(self.actions)

    def __iter__(self) -> Action:
        for action in self.actions:
            yield action

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __getitem__(self, index):
        if self.action_dim == 0:
            return []  # no action vector
        return self.actions[index]  # action vector

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def action_dim(self):
        try:
            return len(self.actions[0])
        except TypeError:
            return 0  # indicate that init with action index

    # pyre-fixme[3]: Return type must be annotated.
    def to_tensor(self):
        if self.action_dim == 0:
            return torch.zeros(self.n, 0)
        return torch.Tensor(self.actions)

    def cat_state_tensor(self, subjective_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subjective_state: a tensor representing subjective state of shape (batch_size, state_dim) or (state_dim)
        Returns:
            cat subjective_state to every action and return a feature tensor with shape (batch_size, action_count, feature_dim)
        """
        # TODO: this method should not be here because a discrete action space shouldn't know about states,
        # it should simply know about the discrete set of actions.
        action_dim = self.action_dim
        state_dim = subjective_state.shape[-1]
        action_count = self.n

        subjective_state = subjective_state.view(
            -1, state_dim
        )  # reshape to (batch_size, state_dim)
        batch_size = subjective_state.shape[0]

        expanded_state = subjective_state.unsqueeze(1).repeat(
            1, action_count, 1
        )  # expand to (batch_size, action_count, state_dim)

        actions = self.to_tensor().to(
            subjective_state.device
        )  # (action_count, action_dim)
        expanded_action = actions.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # batch_size, action_count, action_dim
        new_feature = torch.cat(
            [expanded_state, expanded_action], dim=2
        )  # batch_size, action_count, feature_dim

        torch._assert(
            new_feature.shape == (batch_size, action_count, state_dim + action_dim),
            "The shape of the concatenated feature is wrong. Expected "
            f"{(batch_size, action_count, state_dim + action_dim)}, got {new_feature.shape}",
        )
        return new_feature.to(subjective_state.device)
