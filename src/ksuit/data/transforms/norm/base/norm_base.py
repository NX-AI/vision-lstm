from ksuit.data.transforms.base import Transform


class NormBase(Transform):
    def __init__(self, inverse=False, inplace=True, **kwargs):
        super().__init__(**kwargs)
        self.inverse = inverse
        self.inplace = inplace

    def __call__(self, x, ctx=None):
        if self.inverse:
            return self.denormalize(x, inplace=self.inplace)
        return self.normalize(x, inplace=self.inplace)

    def normalize(self, x, inplace=True):
        raise NotImplementedError

    def denormalize(self, x, inplace=True):
        raise NotImplementedError
