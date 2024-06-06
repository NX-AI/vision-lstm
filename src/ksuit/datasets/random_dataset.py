import torch

from ksuit.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, length=10, x_shape=(7,), num_classes=9, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.x_shape = x_shape
        self.num_classes = num_classes
        rng = torch.Generator().manual_seed(0)
        self.x = torch.randn(length, *x_shape, generator=rng)
        self.classes = torch.randint(num_classes, size=(length,), generator=rng)

    def getshape_x(self):
        return self.x_shape

    def getshape_class(self):
        return self.num_classes,

    def getitem_x(self, idx):
        return self.x[idx].clone()

    def getitem_class(self, idx):
        return self.classes[idx].clone()

    def __getitem__(self, idx):
        return self.getitem_x(idx), self.getitem_class(idx)

    def __len__(self):
        return self.length
