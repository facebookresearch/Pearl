import torch
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)


class IdentityActionRepresentationModule(ActionRepresentationModule):
    """
    An trivial class that outputs actions identitically as input.
    """

    def __init__(self, max_number_actions: int) -> None:
        super(IdentityActionRepresentationModule, self).__init__()
        self._max_number_actions = max_number_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def max_number_actions(self) -> int:
        return self._max_number_actions
