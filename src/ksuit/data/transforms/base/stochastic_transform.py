import torch

from ksuit.data.utils.random import get_rng_from_global
from .transform import Transform


class StochasticTransform(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = None
        self._is_worker_rng = False

    @property
    def rng(self):
        # if main process initializes the rng, it will be copied to the worker processes
        # this leads to all worker processes getting the same rng state
        # to avoid this, recreate the rng when it is retrieved for the first time from a worker process
        if torch.utils.data.get_worker_info() is not None:
            # initialize rng in worker process
            if self._rng is None or not self._is_worker_rng:
                self._rng = get_rng_from_global()
                self._is_worker_rng = True
        elif self._rng is None:
            # initialize rng in main process
            self._rng = get_rng_from_global()
        return self._rng

    @rng.setter
    def rng(self, value):
        # if rng is set from worker_process (e.g. via worker_init_fn or XTransformWrapper) dont overwrite it in getter
        if torch.utils.data.get_worker_info() is not None:
            self._is_worker_rng = True
        self._rng = value

    @property
    def is_deterministic(self):
        return False

    def set_rng(self, rng):
        self.rng = rng
        return self

    def worker_init_fn(self, *args, **kwargs):
        super().worker_init_fn(*args, **kwargs)
        # problem: since rngs are initialized in the __init__ methods they are copied when workers are spawned
        # solution: overwrite the rng when workers are spawned
        self.set_rng(get_rng_from_global())

    def __call__(self, x, ctx=None):
        raise NotImplementedError
