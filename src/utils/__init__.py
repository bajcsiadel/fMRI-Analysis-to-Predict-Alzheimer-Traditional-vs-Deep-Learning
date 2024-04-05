import torch


def get_device(disable_gpu):
    import platform

    if disable_gpu:
        return "cpu"

    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    elif platform.system() in ["Linux", "Windows"] and torch.cuda.is_available():
        return "cuda"

    return "cpu"
