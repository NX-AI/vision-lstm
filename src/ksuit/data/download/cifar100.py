import os

from torchvision.datasets import CIFAR100

from ksuit.utils.param_checking import to_path


def download_cifar100(root):
    root = to_path(root, mkdir=True)
    if (root / "cifar-100-python").exists():
        print(f"dataset was already downloaded -> skip ('{root.as_posix()}')")
        return
    # download
    _ = CIFAR100(root=root, train=True, download=True)
    _ = CIFAR100(root=root, train=False, download=True)
    # delete zip
    os.remove(root / "cifar-100-python.tar.gz")
