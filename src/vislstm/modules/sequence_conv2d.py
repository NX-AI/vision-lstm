import math

import einops
from torch import nn


class SequenceConv2d(nn.Conv2d):
    def forward(self, x):
        assert x.ndim == 3
        h = math.sqrt(x.size(1))
        assert h.is_integer()
        h = int(h)
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        x = super().forward(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x
