import torch


class MeanFinalizer:
    def __call__(self, features: list):
        return torch.stack(features).mean(dim=0)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__
