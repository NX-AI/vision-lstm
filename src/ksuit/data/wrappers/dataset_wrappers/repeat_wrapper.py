import numpy as np

from ksuit.data import Subset


class RepeatWrapper(Subset):
    """ repeats the dataset <repetitions> times or until it reaches <min_size>"""

    def __init__(self, dataset, repetitions=None, min_size=None):
        assert (repetitions is not None) ^ (min_size is not None)
        assert len(dataset) > 0
        self.repetitions = repetitions
        self.min_size = min_size

        if min_size is not None:
            assert isinstance(min_size, int) and min_size > 0
            self.repetitions = int(np.ceil(min_size / len(dataset)))
        else:
            assert repetitions > 0

        # repeat indices <repetitions> times in round-robin fashion (indices are like [0, 1, 2, 0, 1, 2])
        indices = np.tile(np.arange(len(dataset), dtype=np.int64), self.repetitions)
        super().__init__(dataset=dataset, indices=indices)
