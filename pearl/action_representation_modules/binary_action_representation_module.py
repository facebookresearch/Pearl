import torch

from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)


class BinaryActionTensorRepresentationModule(ActionRepresentationModule):
    """
    Transform index to its binary representation.
    """

    # TODO: replace max_actions with action_space.n after action
    # space standardization
    def __init__(self, bits_num: int) -> None:
        super(BinaryActionTensorRepresentationModule, self).__init__()
        self.bits_num = bits_num
        self._max_number_actions: int = 2**bits_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.binary(x, self.bits_num)
        # (batch_size x action_dim)

    def binary(self, x: torch.Tensor, bits_num: float) -> torch.Tensor:
        mask = 2 ** torch.arange(bits_num).to(device=x.device)
        x = x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
        return x.to(dtype=torch.float32)

    @property
    def max_number_actions(self) -> int:
        return self._max_number_actions
