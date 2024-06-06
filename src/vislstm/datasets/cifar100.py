from ksuit.data import Dataset
from ksuit.providers import DatasetConfigProvider


class Cifar100(Dataset):
    def __init__(
            self,
            split,
            global_root=None,
            local_root=None,
            dataset_config_provider: DatasetConfigProvider = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if dataset_config_provider is not None:
            global_root, _ = dataset_config_provider.get_roots(global_root=global_root, identifier="cifar100")
        if local_root is not None:
            self.logger.info(f"cifar100 is an in-memory dataset -> local_root is ignored")
        self.global_root = global_root
        assert split in ["train", "test"]
        self.split = split

        from torchvision.datasets import CIFAR100
        train = split == "train"
        self.dataset = CIFAR100(root=global_root, train=train, download=False)

    def __str__(self):
        return f"{type(self).__name__}(split={self.split})"

    @staticmethod
    def getshape_class():
        return 100,

    def getitem_x(self, idx):
        return self.dataset[idx][0]

    def getitem_class(self, idx):
        return self.dataset.targets[idx]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
