import torch
import torchvision.transforms.functional as F

from .base import NormBase


class ImageMomentNorm(NormBase):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def normalize(self, x, inplace=True):
        if not torch.is_tensor(x):
            x = F.to_tensor(x)
        return F.normalize(x, mean=self.mean, std=self.std, inplace=inplace)

    def denormalize(self, x, inplace=True):
        if not torch.is_tensor(x):
            x = F.to_tensor(x)
        inv_std = tuple(1. / std for std in self.std)
        inv_mean = tuple(-mean for mean in self.mean)
        zero = tuple(0. for _ in self.mean)
        one = tuple(1. for _ in self.std)
        x = F.normalize(x, mean=zero, std=inv_std, inplace=inplace)
        return F.normalize(x, mean=inv_mean, std=one, inplace=inplace)
