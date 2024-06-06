from .base.pooling_base import PoolingBase


class MeanPatch(PoolingBase):
    def get_output_shape(self, input_shape):
        _, dim = input_shape
        return dim,

    def forward(self, all_tokens, *_, **__):
        if "num_aux_tokens" in self.static_ctx:
            num_cls_tokens = self.static_ctx["num_aux_tokens"]
        elif "num_cls_tokens" in self.static_ctx:
            num_cls_tokens = self.static_ctx["num_cls_tokens"]
        else:
            raise NotImplementedError
        return all_tokens[:, num_cls_tokens:].mean(dim=1)

    def __str__(self):
        return type(self).__name__
