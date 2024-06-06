from torch import nn


# derive from nn.Module to be usable in nn.Sequential
class PoolingBase(nn.Module):
    def __init__(self, static_ctx=None):
        super().__init__()
        self.static_ctx = static_ctx

    def get_output_shape(self, input_shape):
        raise NotImplementedError

    def register_hooks(self, model):
        pass

    def enable_hooks(self):
        pass

    def disable_hooks(self):
        pass

    def clear_extractor_outputs(self):
        pass

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(str(self))

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError
