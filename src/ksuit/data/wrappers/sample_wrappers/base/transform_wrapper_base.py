import numpy as np

from ksuit.data import Wrapper
from ksuit.data.transforms import Transform
from ksuit.data.utils.optional_kwargs import optional_ctx
from ksuit.factory import MasterFactory


class TransformWrapperBase(Wrapper):
    def __init__(self, dataset, transform, seed=None):
        super().__init__(dataset=dataset)
        self.transform = MasterFactory.get("transform").create(transform)
        self.seed = seed

    def _getitem(self, item, idx, ctx=None):
        if self.seed is not None:
            rng = np.random.default_rng(seed=self.seed + idx)
            if isinstance(self.transform, Transform):
                self.transform.set_rng(rng)
        return self.transform(item, **optional_ctx(fn=self.transform.__call__, ctx=ctx))

    def worker_init_fn(self, *args, **kwargs):
        super().worker_init_fn(*args, **kwargs)
        if isinstance(self.transform, Transform):
            self.transform.worker_init_fn(*args, **kwargs)
