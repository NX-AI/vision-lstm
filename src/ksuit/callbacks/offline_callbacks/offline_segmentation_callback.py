from functools import partial
from itertools import product

import torch.nn.functional as F
from torchvision.transforms.functional import hflip

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.utils.miou_utils import intersect_and_union, total_area_to_metrics


class OfflineSegmentationCallback(PeriodicCallback):
    def __init__(self, dataset_key, mode, ignore_index=-1, mode_kwargs=None, interpolation="bicubic", **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.ignore_index = ignore_index
        self.mode = mode
        self.interpolation = interpolation
        self.mode_kwargs = mode_kwargs or {}
        self.__config_id = None

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode="x segmentation")

    def _forward(self, batch, model, trainer):
        x = batch["x"]
        target = batch["segmentation"]
        x = x.to(model.device, non_blocking=True)
        target = target.to(model.device, non_blocking=True)

        # resize short side to model input
        og_shape = x.shape

        if self.mode == "slide":
            x = F.interpolate(x, size=min(model.input_shape[1:]), mode=self.interpolation)
            assert len(x) == 1, f"slide inference requires batch_size=1"
            batch_size = 1
            h_stride, w_stride = self.mode_kwargs["stride"]
            h_crop, w_crop = model.input_shape[1:]
            assert h_stride <= h_crop
            assert w_stride <= w_crop
            _, _, h_img, w_img = x.shape
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            pred = x.new_zeros((batch_size, model.output_shape[0], h_img, w_img))
            count = x.new_zeros((batch_size, 1, h_img, w_img))
            for h_idx, w_idx in product(range(h_grids), range(w_grids)):
                h_start = h_idx * h_stride
                w_start = w_idx * w_stride
                h_end = min(h_start + h_crop, h_img)
                w_end = min(w_start + w_crop, w_img)
                h_start = max(h_end - h_crop, 0)
                w_start = max(w_end - w_crop, 0)
                crop_img = x[:, :, h_start:h_end, w_start:w_end]
                # pad if image is too small
                pad_h = h_crop - crop_img.size(2)
                pad_w = w_crop - crop_img.size(3)
                crop_img = F.pad(crop_img, (0, pad_w, 0, pad_h))

                with trainer.autocast_context:
                    logits = model.segment(crop_img)
                cutoff_h = crop_img.size(2) - pad_h
                cutoff_w = crop_img.size(3) - pad_w
                pred[:, :, h_start:h_end, w_start:w_end] += logits[:, :, :cutoff_h, :cutoff_w]
                count[:, :, h_start:h_end, w_start:w_end] += 1
            #
            assert (count == 0).sum() == 0
            pred /= count
            # resize back to original resolution (https://arxiv.org/abs/2404.12172)
            pred = F.interpolate(pred, size=og_shape[2:], mode="bilinear")
        elif self.mode == "multiscale":
            assert len(x) == 1, f"multiscale inference requires batch_size=1"
            og_x = x
            batch_size = 1
            scale_factors = self.mode_kwargs["scale_factors"]
            flip = self.mode_kwargs.get("flip", False)
            if flip:
                flip_options = [False, True]
            else:
                flip_options = [False]
            _, _, h_og, w_og = og_shape
            pred = x.new_zeros((batch_size, model.output_shape[0], h_og, w_og))
            for flip_option, scale_factor in product(flip_options, scale_factors):
                x = og_x
                if flip_option:
                    x = hflip(x)
                x = F.interpolate(x, scale_factor=scale_factor, mode=self.interpolation)
                h_stride, w_stride = self.mode_kwargs["stride"]
                h_crop, w_crop = model.input_shape[1:]
                assert h_stride <= h_crop
                assert w_stride <= w_crop
                _, _, h_img, w_img = x.shape
                h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
                cur_pred = x.new_zeros((batch_size, model.output_shape[0], h_img, w_img))
                count = x.new_zeros((batch_size, 1, h_img, w_img))
                for h_idx, w_idx in product(range(h_grids), range(w_grids)):
                    h_start = h_idx * h_stride
                    w_start = w_idx * w_stride
                    h_end = min(h_start + h_crop, h_img)
                    w_end = min(w_start + w_crop, w_img)
                    h_start = max(h_end - h_crop, 0)
                    w_start = max(w_end - w_crop, 0)
                    crop_img = x[:, :, h_start:h_end, w_start:w_end]
                    # pad if image is too small
                    pad_h = h_crop - crop_img.size(2)
                    pad_w = w_crop - crop_img.size(3)
                    crop_img = F.pad(crop_img, (0, pad_w, 0, pad_h))

                    with trainer.autocast_context:
                        logits = model.segment(crop_img)
                    cutoff_h = crop_img.size(2) - pad_h
                    cutoff_w = crop_img.size(3) - pad_w
                    cur_pred[:, :, h_start:h_end, w_start:w_end] += logits[:, :, :cutoff_h, :cutoff_w]
                    count[:, :, h_start:h_end, w_start:w_end] += 1
                #
                assert (count == 0).sum() == 0
                cur_pred /= count
                # resize back to original resolution
                cur_pred = F.interpolate(cur_pred, size=(h_og, w_og), mode="bilinear")
                if flip_option:
                    cur_pred = hflip(cur_pred)
                pred += cur_pred
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented")

        # metrics
        pred = pred.argmax(dim=1)
        result = intersect_and_union(
            pred=pred,
            target=target,
            num_classes=model.output_shape[0],
            ignore_index=self.ignore_index,
        )
        return result

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        # iterate
        result = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
        # sum results over batch dimension
        # area_intersect -> total_area_intersect
        # area_union -> total_area_union
        # area_pred_label -> totalarea_pred_label
        # area_label -> total_area_label
        result = {f"total_{key}": value.sum(dim=0) for key, value in result.items()}

        # calculate metrics
        metrics = total_area_to_metrics(**result)
        for key, value in metrics.items():
            self.writer.add_scalar(
                f"{key}/{self.dataset_key}/main",
                value.mean(),
                logger=self.logger,
                format_str=".6f",
            )
