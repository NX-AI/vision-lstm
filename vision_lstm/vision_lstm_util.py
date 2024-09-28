import collections
import itertools
import math

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(itertools.repeat(x, n))

    return parse


# adapted from timm (timm/models/layers/helpers.py)
def to_ntuple(x, n):
    return _ntuple(n=n)(x)


# from kappamodules.functional.pos_embed import interpolate_sincos
def interpolate_sincos(embed, seqlens, mode="bicubic"):
    assert embed.ndim - 2 == len(seqlens)
    embed = F.interpolate(
        einops.rearrange(embed, "1 ... dim -> 1 dim ..."),
        size=seqlens,
        mode=mode,
    )
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed


class SequenceConv2d(nn.Conv2d):
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens

    def forward(self, x):
        assert x.ndim == 3
        if self.seqlens is None:
            # assuming square input
            h = math.sqrt(x.size(1))
            assert h.is_integer()
            h = int(h)
        else:
            assert len(self.seqlens) == 2
            h = self.seqlens[0]
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        x = super().forward(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x


# from kappamodules.vit import VitPatchEmbed
class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, stride=None, init_weights="xavier_uniform"):
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        if stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(stride, n=self.ndim)
        for i in range(self.ndim):
            assert resolution[i] % self.patch_size[i] == 0, \
                f"resolution[{i}] % patch_size[{i}] != 0 (resolution={resolution} patch_size={patch_size})"
        self.seqlens = [resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        if self.patch_size == self.stride:
            # use primitive type as np.prod gives np.int which is not compatible with all serialization/logging
            self.num_patches = int(np.prod(self.seqlens))
        else:
            if self.ndim == 1:
                conv_func = F.conv1d
            elif self.ndim == 2:
                conv_func = F.conv2d
            elif self.ndim == 3:
                conv_func = F.conv3d
            else:
                raise NotImplementedError
            self.num_patches = conv_func(
                input=torch.zeros(1, 1, *resolution),
                weight=torch.zeros(1, 1, *self.patch_size),
                stride=self.stride,
            ).numel()

        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError

        self.proj = conv_ctor(num_channels, dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            # initialize as nn.Linear
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        assert all(x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)), \
            f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        x = self.proj(x)
        x = einops.rearrange(x, "b c ... -> b ... c")
        return x


# from kappamodules.vit import VitPosEmbed2d
class VitPosEmbed2d(nn.Module):
    def __init__(self, seqlens, dim: int, allow_interpolation: bool = True):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.allow_interpolation = allow_interpolation
        self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == self._expected_x_ndim
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.allow_interpolation
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[1:-1])
        else:
            embed = self.embed
        return x + embed


# from kappamodules.layers import DropPath
class DropPath(nn.Sequential):
    """
    Efficiently drop paths (Stochastic Depth) per sample such that dropped samples are not processed.
    This is a subclass of nn.Sequential and can be used either as standalone Module or like nn.Sequential.
    Examples::
        >>> # use as nn.Sequential module
        >>> sequential_droppath = DropPath(nn.Linear(4, 4), drop_prob=0.2)
        >>> y = sequential_droppath(torch.randn(10, 4))

        >>> # use as standalone module
        >>> standalone_layer = nn.Linear(4, 4)
        >>> standalone_droppath = DropPath(drop_prob=0.2)
        >>> y = standalone_droppath(torch.randn(10, 4), standalone_layer)
    """

    def __init__(
            self,
            *args,
            drop_prob: float = 0.,
            scale_by_keep: bool = True,
            stochastic_drop_prob: bool = False,
            drop_prob_tolerance: float = 0.01,
    ):
        super().__init__(*args)
        assert 0. <= drop_prob < 1.
        self._drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob
        self.drop_prob_tolerance = drop_prob_tolerance

    @property
    def drop_prob(self):
        return self._drop_prob

    @drop_prob.setter
    def drop_prob(self, value):
        assert 0. <= value < 1.
        self._drop_prob = value

    @property
    def keep_prob(self):
        return 1. - self.drop_prob

    def forward(self, x, residual_path=None, residual_path_kwargs=None):
        assert (len(self) == 0) ^ (residual_path is None)
        residual_path_kwargs = residual_path_kwargs or {}
        if self.drop_prob == 0. or not self.training:
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs)
            else:
                return x + residual_path(x, **residual_path_kwargs)
        bs = len(x)
        # for small batchsizes its not possible to do it efficiently
        # e.g. batchsize 2 with drop_rate=0.05 would drop 1 sample and therefore increase the drop_rate to 0.5
        # resolution: fall back to inefficient version
        keep_count = max(int(bs * self.keep_prob), 1)
        # allow some level of tolerance
        actual_keep_prob = keep_count / bs
        drop_path_delta = self.keep_prob - actual_keep_prob
        # if drop_path_delta > self.drop_prob_tolerance:
        #     warnings.warn(
        #         f"efficient stochastic depth (DropPath) would change drop_path_rate by {drop_path_delta:.4f} "
        #         f"because the batchsize is too small to accurately drop {bs - keep_count} samples per forward pass"
        #         f" -> forcing stochastic_drop_prob=True drop_path_rate={self.drop_prob}"
        #     )

        # inefficient drop_path
        if self.stochastic_drop_prob or drop_path_delta > self.drop_prob_tolerance:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(self.keep_prob)
            if self.scale_by_keep:
                random_tensor.div_(self.keep_prob)
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs) * random_tensor
            else:
                return x + residual_path(x, **residual_path_kwargs) * random_tensor

        # generate indices to keep (propagated through transform path)
        scale = bs / keep_count
        perm = torch.randperm(bs, device=x.device)[:keep_count]

        # propagate
        if self.scale_by_keep:
            alpha = scale
        else:
            alpha = 1.
        # reduce kwargs (e.g. used for DiT block where scale/shift/gate is passed and also has to be reduced)
        residual_path_kwargs = {
            key: value[perm] if torch.is_tensor(value) else value
            for key, value in residual_path_kwargs.items()
        }
        if residual_path is None:
            residual = super().forward(x[perm], **residual_path_kwargs)
        else:
            residual = residual_path(x[perm], **residual_path_kwargs)
        return torch.index_add(
            x.flatten(start_dim=1),
            dim=0,
            index=perm,
            source=residual.to(x.dtype).flatten(start_dim=1),
            alpha=alpha,
        ).view_as(x)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'