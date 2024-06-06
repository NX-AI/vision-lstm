import torch


class ConcatFinalizer:
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, features: list):
        return torch.concat(features, dim=self.dim)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{type(self).__name__}(dim={self.dim})"
