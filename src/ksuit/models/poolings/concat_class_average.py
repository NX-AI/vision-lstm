import torch

from .base.pooling_base import PoolingBase


class ConcatClassAverage(PoolingBase):
    def get_output_shape(self, input_shape):
        _, dim = input_shape
        return 2 * dim,

    def forward(self, all_tokens, *_, **__):
        assert self.static_ctx["num_aux_tokens"] == 1
        cls = all_tokens[:, 0]
        avg = all_tokens[:, self.static_ctx["num_aux_tokens"]:].mean(dim=1)
        return torch.concat([cls, avg], dim=1)

    def __str__(self):
        return type(self).__name__
