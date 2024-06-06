from ksuit.models import PoolingBase


class FirstToken(PoolingBase):
    def get_output_shape(self, input_shape):
        _, dim = input_shape
        return dim,

    def forward(self, all_tokens, *_, **__):
        return all_tokens[:, 0]

    def __str__(self):
        return type(self).__name__
