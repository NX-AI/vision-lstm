from .base.pooling_base import PoolingBase


class ClassToken(PoolingBase):
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
        location = self.static_ctx.get("cls_token_location", "first")
        if location == "first":
            x = all_tokens[:, :num_cls_tokens]
        elif location == "middle":
            middle = all_tokens.size(1) // 2
            half_num_tokens = num_cls_tokens // 2
            start = middle - half_num_tokens
            end = start + num_cls_tokens
            x = all_tokens[:, start:end]
        else:
            raise NotImplementedError
        return x.flatten(start_dim=1)

    def __str__(self):
        return type(self).__name__
