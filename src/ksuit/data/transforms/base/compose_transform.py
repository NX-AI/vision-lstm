from ksuit.data.utils.optional_kwargs import optional_ctx
from ksuit.factory import MasterFactory
from .transform import Transform


class ComposeTransform(Transform):
    def __init__(self, transforms):
        super().__init__()
        factory = MasterFactory.get("transform")
        self.transforms = [factory.create(transform) for transform in transforms]

    def worker_init_fn(self, *args, **kwargs):
        super().worker_init_fn(*args, **kwargs)
        for t in self.transforms:
            if isinstance(t, Transform):
                t.worker_init_fn(*args, **kwargs)

    @property
    def is_deterministic(self):
        return all(t.is_deterministic for t in self.transforms)

    def __call__(self, x, ctx=None):
        if ctx is None:
            ctx = {}
        for t in self.transforms:
            # apply to one sample
            x = t(x, **optional_ctx(fn=t.__call__, ctx=ctx))
        return x

    def set_rng(self, rng):
        for t in self.transforms:
            if isinstance(t, Transform):
                t.set_rng(rng)
        return self
