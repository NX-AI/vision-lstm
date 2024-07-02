import einops
import numpy as np
import torch

from .base.pooling_base import PoolingBase


class ToImage(PoolingBase):
    def __init__(self, concat_cls=False, **kwargs):
        super().__init__(**kwargs)
        self.concat_cls = concat_cls

    def get_output_shape(self, input_shape):
        num_tokens, dim = input_shape
        num_patch_tokens = num_tokens - self.static_ctx["num_aux_tokens"]
        assert num_patch_tokens == np.prod(self.static_ctx["sequence_lengths"])
        if self.concat_cls:
            dim *= 2
        return dim, *self.static_ctx["sequence_lengths"]

    def forward(self, all_tokens, *_, **__):
        if "num_aux_tokens" in self.static_ctx:
            num_cls_tokens = self.static_ctx["num_aux_tokens"]
        elif "num_cls_tokens" in self.static_ctx:
            num_cls_tokens = self.static_ctx["num_cls_tokens"]
        else:
            raise NotImplementedError
        patch_tokens = all_tokens[:, num_cls_tokens:]
        seqlen_h, seqlen_w = self.static_ctx["sequence_lengths"]
        img = einops.rearrange(
            patch_tokens,
            "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
            seqlen_h=seqlen_h,
            seqlen_w=seqlen_w,
        )
        if self.concat_cls:
            assert self.static_ctx["num_aux_tokens"] == 1
            cls_tokens = einops.repeat(
                all_tokens[:, :1],
                "b 1 dim -> b dim seqlen_h seqlen_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
            img = torch.concat([cls_tokens, img], dim=1)
        return img

    def __str__(self):
        return type(self).__name__
