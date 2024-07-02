from collections import defaultdict
from functools import partial
from itertools import product

import numpy as np
from kappadata.common.transforms import ImagenetNoaugTransform
from torchmetrics.functional.classification import multiclass_accuracy

from vislstm.datasets.imagenet_corruption import ImagenetCorruption
from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.data.wrappers import XTransformWrapper


class OfflineImagenetCorruptionCallback(PeriodicCallback):
    def __init__(self, resize_size=224, center_crop_size=224, interpolation="bicubic", **kwargs):
        super().__init__(**kwargs)
        self.transform = ImagenetNoaugTransform(
            resize_size=resize_size,
            center_crop_size=center_crop_size,
            interpolation=interpolation,
        )
        self.dataset_keys = [
            f"imagenet_c_{distortion}_{level}"
            for distortion, level in product(
                [
                    # blur
                    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                    # digitarl
                    "contrast", "elastic_transform", "jpeg_compression", "pixelate",
                    # extra
                    "gaussian_blur", "saturate", "spatter", "speckle_noise",
                    # noise
                    "gaussian_noise", "shot_noise", "impulse_noise",
                    # weather
                    "frost", "snow", "fog", "brightness",
                ],
                [1, 2, 3, 4, 5],
            )
        ]
        self.__config_ids = {}
        self.n_classes = None

    def _before_training(self, model, **kwargs):
        assert len(model.output_shape) == 1
        self.n_classes = self.data_container.get_dataset("train").getdim_class()

    def register_root_datasets(self, dataset_config_provider=None, is_mindatarun=False):
        if is_mindatarun:
            raise NotImplementedError
        for key in self.dataset_keys:
            if key in self.data_container.datasets:
                continue
            temp = key.replace("imagenet_c_", "")
            distortion = temp[:-2]
            level = temp[-1]
            dataset = ImagenetCorruption(
                split=f"{distortion}/{level}",
                dataset_config_provider=dataset_config_provider,
            )
            dataset = XTransformWrapper(dataset=dataset, transform=ImagenetNoaugTransform())
            assert len(dataset) == 50000
            self.data_container.datasets[key] = dataset

    def _register_sampler_configs(self, trainer):
        for key in self.dataset_keys:
            self.__config_ids[key] = self._register_sampler_config_from_key(key=key, mode="x class")

    @staticmethod
    def _forward(batch, model, trainer):
        x = batch["x"]
        cls = batch["class"]
        x = x.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            predictions = model.classify(x)
        predictions = {name: prediction.cpu() for name, prediction in predictions.items()}
        return predictions, cls.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        all_accuracies = defaultdict(dict)
        for dataset_key in self.dataset_keys:
            # extract
            predictions, classes = self.iterate_over_dataset(
                forward_fn=partial(self._forward, model=model, trainer=trainer),
                config_id=self.__config_ids[dataset_key],
                batch_size=batch_size,
                data_iter=data_iter,
            )

            # push to GPU for accuracy calculation
            predictions = {k: v.to(model.device, non_blocking=True) for k, v in predictions.items()}
            classes = classes.to(model.device, non_blocking=True)

            # log
            for name, prediction in predictions.items():
                acc = multiclass_accuracy(
                    preds=prediction,
                    target=classes,
                    num_classes=self.n_classes,
                    average="micro",
                ).item()
                self.writer.add_scalar(f"accuracy1/{dataset_key}/{name}", acc, logger=self.logger, format_str=".4f")
                all_accuracies[name][dataset_key] = acc

        # summarize over all
        for name in all_accuracies.keys():
            acc = float(np.mean(list(all_accuracies[name].values())))
            self.writer.add_scalar(f"accuracy1/imagenet_c_overall/{name}", acc, logger=self.logger, format_str=".4f")
