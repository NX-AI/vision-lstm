from .base.pooling_base import PoolingBase


class IndexPooling(PoolingBase):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def get_output_shape(self, input_shape):
        _, dim = input_shape
        return dim,

    def forward(self, all_tokens, *_, **__):
        return all_tokens[:, self.index]

    def __str__(self):
        return f"{type(self).__name__}(index={self.index})"
