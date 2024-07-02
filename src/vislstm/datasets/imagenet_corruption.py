from ksuit.datasets import MulticlassDataset
from ksuit.providers import DatasetConfigProvider


class ImagenetCorruption(MulticlassDataset):
    def __init__(
            self,
            global_root=None,
            local_root=None,
            dataset_config_provider: DatasetConfigProvider = None,
            **kwargs,
    ):
        if dataset_config_provider is not None:
            global_root, local_root = dataset_config_provider.get_roots(
                global_root=global_root,
                local_root=local_root,
                identifier="imagenet_c",
            )
            if local_root is not None:
                local_root = local_root / "imagenet_c"
        super().__init__(global_root=global_root, local_root=local_root, **kwargs)

    def getall_class(self):
        return self.dataset.targets