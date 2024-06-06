from torchvision.datasets.folder import DatasetFolder

from ksuit.data import Dataset
from ksuit.data.copy.multiclass_dataset import multiclass_global_to_local
from ksuit.data.loaders import fname_to_loader
from ksuit.utils.num_worker_heuristic import get_fair_cpu_count
from ksuit.utils.param_checking import to_path
from ksuit.distributed import is_local_rank0, barrier

class MulticlassDataset(Dataset):
    def __init__(
            self,
            global_root,
            local_root=None,
            split=None,
            loader=None,
            is_valid_file=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_root = to_path(global_root)
        self.local_root = to_path(local_root)
        if split is not None:
            self.global_root = self.global_root / split
            if self.local_root is not None:
                self.local_root = self.local_root / split
        if self.local_root is None:
            self.source_root = self.global_root
        else:
            self.source_root = self.local_root
            if is_local_rank0():
                multiclass_global_to_local(
                    global_root=self.global_root,
                    local_root=self.local_root,
                    num_workers=min(10, get_fair_cpu_count()),
                    log_fn=self.logger.info,
                )
            barrier()

        # init dataset (loader is set afterwards based on found file endings)
        self.dataset = DatasetFolder(
            root=self.source_root,
            is_valid_file=is_valid_file or (lambda _: True),
            loader=None,
        )

        # set loader
        if loader is None:
            self.dataset.loader = fname_to_loader(self.dataset.samples[0][0])
        else:
            self.dataset.loader = loader

    def getitem_x(self, idx):
        x, _ = self.dataset[idx]
        return x

    def getitem_class(self, idx):
        return self.dataset.targets[idx]

    def getshape_class(self):
        return len(self.dataset.classes),

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
