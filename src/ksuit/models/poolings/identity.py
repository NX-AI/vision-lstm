from .base.pooling_base import PoolingBase


class Identity(PoolingBase):
    def get_output_shape(self, input_shape):
        return input_shape

    def forward(self, all_tokens, *_, **__):
        return all_tokens

    def __str__(self):
        return type(self).__name__
