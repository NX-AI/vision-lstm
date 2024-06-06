from functools import partial
from itertools import product

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.data.wrappers import XTransformWrapper
from ksuit.utils.tensor_hashing import hash_rgb
from ksuit.utils.transform_utils import get_denorm_transform


class VisualizeSegmentationCallback(PeriodicCallback):
    def __init__(self, dataset_key, mode, mode_kwargs=None, save_input=False, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.mode = mode
        self.mode_kwargs = mode_kwargs or {}
        self.save_input = save_input
        self.__config_id = None
        self.out = self.path_provider.stage_output_path / "segmentation"
        self.out.mkdir()

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode="index x segmentation")

    def _before_training(self, **kwargs):
        # extract normalization transform
        if self.save_input:
            wrapper = self.data_container.get_dataset(self.dataset_key).get_wrapper_of_type(XTransformWrapper)
            self.denormalize = get_denorm_transform(wrapper.transform)
        else:
            self.denormalize = None

    def _forward(self, batch, model, trainer):
        index = batch["index"]
        x = batch["x"]
        target = batch["segmentation"]
        x = x.to(model.device, non_blocking=True)
        target = target.to(model.device, non_blocking=True)

        if self.mode == "slide":
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
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented")
        # convert to rgb
        pred = hash_rgb(pred.argmax(dim=1))
        target = hash_rgb(target)

        # store
        for i in range(len(index)):
            idx = index[i].item()
            save_image(
                pred[i],
                self.out / f"{idx:04d}_pred_{self.update_counter.cur_checkpoint}.png",
            )
            target_out = self.out / f"{idx:04d}_gt.png"
            if not target_out.exists():
                save_image(target[i], target_out)
            if self.save_input:
                input_out = self.out / f"{idx:04d}_src.png"
                if not input_out.exists():
                    save_image(self.denormalize(x[i]), input_out)

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        # iterate
        self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
