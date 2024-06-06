import torch


class StackFinalizer:
    def __init__(self, dim: int = 1):
        self.dim = dim

    def __call__(self, features: list):
        return torch.stack(features, dim=self.dim)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__
