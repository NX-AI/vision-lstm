import os

from torchvision.datasets import CIFAR10

from ksuit.utils.param_checking import to_path


def download_cifar10(root):
    root = to_path(root, mkdir=True)
    if (root / "cifar-10-python").exists():
        print(f"dataset was already downloaded -> skip ('{root.as_posix()}')")
        return
    # download
    _ = CIFAR10(root=root, train=True, download=True)
    _ = CIFAR10(root=root, train=False, download=True)
    # delete zip
    os.remove(root / "cifar-10-python.tar.gz")
