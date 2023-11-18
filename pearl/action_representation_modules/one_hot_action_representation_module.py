import torch
import torch.nn.functional as F

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)


class OneHotActionTensorRepresentationModule(ActionRepresentationModule):
    """
    An one-hot action representation module.
    """

    # TODO: replace max_actions with action_space.n after action
    # space standardization
    def __init__(self, max_actions: int) -> None:
        super(OneHotActionTensorRepresentationModule, self).__init__()
        self.max_actions = max_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            x.long(), num_classes=self.max_actions
        )  # (batch_size x action_dim)
