from torch import Tensor

# An `Action` is expected to be a 1-dim Tensor of shape `(d,)`, where `d` is the
# is the dimensionality of the action.
Action = Tensor
