import torch

pearl_device = None


def get_pearl_device():
    global pearl_device
    if pearl_device is None:
        pearl_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return pearl_device
