import torch


def get_device(use_cuda=True, verbose=True):
    """
    This function selects the device based on the availability
    and input parameter. After that it returns the device object.
    """

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Currently using: {device}")

    return device
