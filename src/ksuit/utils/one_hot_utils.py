import torch
from torch.nn.functional import one_hot


def to_one_hot_vector(y, n_classes):
    if isinstance(y, int):
        y = torch.tensor(y)
    if y.ndim == 0:
        y = one_hot(y, num_classes=n_classes)
    assert y.ndim == 1
    # one_hot returns int
    return y.float()


def to_one_hot_matrix(y, n_classes):
    if y.ndim == 1:
        y = one_hot(y, num_classes=n_classes)
    assert y.ndim == 2
    # one_hot returns int
    return y.float()
