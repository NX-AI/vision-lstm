from ksuit.models import PoolingBase


class MiddleTokens(PoolingBase):
    def __init__(self, num_tokens=1, aggregate="flatten", **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.aggregate = aggregate

    def get_output_shape(self, input_shape):
        _, dim = input_shape
        if self.aggregate == "flatten":
            return dim * self.num_tokens,
        if self.aggregate == "mean":
            return dim,

    def forward(self, all_tokens, *_, **__):
        middle = all_tokens.size(1) // 2
        half_num_tokens = self.num_tokens // 2
        start = middle - half_num_tokens
        end = start + self.num_tokens
        pooled = all_tokens[:, start:end].flatten(start_dim=1)
        if self.aggregate == "flatten":
            return pooled.flatten(start_dim=1)
        elif self.aggregate == "mean":
            return pooled.mean(dim=1)
        raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}(num_tokens={self.num_tokens},aggregate={self.aggregate})"
