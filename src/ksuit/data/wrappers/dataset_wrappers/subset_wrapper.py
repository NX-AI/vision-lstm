import numpy as np

from ksuit.data.base import Subset


class SubsetWrapper(Subset):
    def __init__(self, dataset, indices=None, start_index=None, end_index=None, start_percent=None, end_percent=None):
        if indices is None:
            if start_index is not None or end_index is not None:
                assert start_percent is None and end_percent is None
                # create indices from start/end index
                assert start_index is None or isinstance(start_index, int)
                assert end_index is None or isinstance(end_index, int)
                end_index = end_index or len(dataset)
                end_index = min(end_index, len(dataset))
                start_index = start_index or 0
                assert start_index <= end_index
                indices = np.arange(start_index, end_index, dtype=np.int64)
            elif start_percent is not None or end_percent is not None:
                # create indices from start/end percent
                assert start_percent is None or (isinstance(start_percent, (float, int)) and 0. <= start_percent <= 1.)
                assert end_percent is None or (isinstance(end_percent, (float, int)) and 0. <= end_percent <= 1.)
                start_percent = start_percent or 0.
                end_percent = end_percent or 1.
                assert start_percent <= end_percent
                start_index = int(start_percent * len(dataset))
                end_index = int(end_percent * len(dataset))
                indices = np.arange(start_index, end_index, dtype=np.int64)
            else:
                raise RuntimeError
        else:
            assert start_index is None and end_index is None
            assert start_percent is None and end_percent is None
            for i in indices:
                assert -len(dataset) <= i < len(dataset)
        super().__init__(dataset=dataset, indices=indices)
