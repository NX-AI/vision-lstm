from ksuit.datasets import MulticlassDataset
from ksuit.providers import DatasetConfigProvider


class ImageNetAdversarial(MulticlassDataset):
    IN1K_TO_INA = [
        6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107,
        108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301,
        306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363,
        372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461,
        462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572,
        573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704,
        719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815,
        820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932,
        933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988,
    ]

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
                identifier="imagenet_a",
            )
            if local_root is not None:
                local_root = local_root / "imagenet_a"
        super().__init__(global_root=global_root, local_root=local_root, split="val", **kwargs)

    def getshape_class(self):
        return 1000,

    @property
    def subset_num_classes(self):
        return 200

    @property
    def class_subset_indices(self):
        return self.IN1K_TO_INA
