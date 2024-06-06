import numpy as np

from ksuit.data.base import Subset


class ShuffleWrapper(Subset):
    def __init__(self, dataset, seed=None):
        self.seed = seed
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random
        indices = np.arange(len(dataset), dtype=np.int64)
        rng.shuffle(indices)
        super().__init__(dataset=dataset, indices=indices)
