from types import MethodType as MethodBoundTo  # alias for greater readability

from pearl.utils.device import get_pearl_device

from torch import nn


class AutoDeviceNNModule(nn.Module):
    """
    An nn.Module that moves itself to get_pearl_device() when its method `forward` is invoked for the first time.
    Afterwards, `forwards` is invoked as usual, without performance penalty for checking if the module has been moved yet.
    """

    # Implementation rationale:
    #
    # To make sure the Module is automatically moved to the device without any special calls on the client side,
    # we could not simply use nn.Module.to(device) at the end of __init__ because that might not move components
    # defined in the __init__ of subclasses, which at that point will not have been run yet.
    #
    # We therefore wait until forward() is invoked for the first time because, by then, we know that all components have been created.
    #
    # However, we do not want the performance penalty of having `forward` checking the device every time it's invoked.
    # Instead, we replace self.forward with a special method `forward_first_run` that does the check and moves
    # the module to the device if needed. This same method then restores `forward` to be the original, check-less method,
    # and, finally, invokes the original forward (since it's meant to be its first run, after all).

    def __init__(self, *args, **kwargs):
        super(AutoDeviceNNModule, self).__init__(*args, **kwargs)
        self.forward = self.forward_first_run

    def forward_first_run(self, *args, **kwargs):
        self.to(get_pearl_device())

        # Restore self.forward to original method
        class_of_self = type(self)
        original_self_forward = class_of_self.forward
        self.forward = MethodBoundTo(original_self_forward, self)

        # invokes original forward
        return self.forward(*args, **kwargs)
