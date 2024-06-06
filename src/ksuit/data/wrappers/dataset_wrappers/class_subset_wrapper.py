import numpy as np

from ksuit.data import Subset


class ClassSubsetWrapper(Subset):
    def __init__(self, dataset, num_classes, seed=0):
        # use numpy for better performance
        all_indices = np.arange(len(dataset), dtype=np.int64)
        classes = np.array(dataset.getall_class())
        total_num_classes = np.max(classes) + 1
        if num_classes < total_num_classes:
            rng = np.random.default_rng(seed=seed)
            valid_classes = np.arange(total_num_classes)
            rng.permutation(valid_classes)
            valid_classes = valid_classes[:num_classes]
            indices = all_indices[np.isin(classes, valid_classes)]
            self.num_classes = num_classes
        else:
            indices = all_indices
            self.num_classes = total_num_classes
        super().__init__(dataset=dataset, indices=indices)

    def getshape_class(self):
        return self.num_classes,