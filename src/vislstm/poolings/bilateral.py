import torch

from ksuit.models import PoolingBase


class Bilateral(PoolingBase):
    def __init__(self, aggregate="flatten", **kwargs):
        super().__init__(**kwargs)
        self.aggregate = aggregate

    def get_output_shape(self, input_shape):
        _, dim = input_shape
        if self.aggregate == "flatten":
            return dim * 2,
        if self.aggregate == "mean":
            return dim,
        raise NotImplementedError

    def forward(self, all_tokens, *_, **__):
        pooled = torch.concat([all_tokens[:, :1], all_tokens[:, -1:]], dim=1)
        if self.aggregate == "flatten":
            return pooled.flatten(start_dim=1)
        elif self.aggregate == "mean":
            return pooled.mean(dim=1)
        raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}(aggregate={self.aggregate})"
