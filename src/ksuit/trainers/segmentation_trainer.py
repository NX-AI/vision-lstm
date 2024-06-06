import torch
import torch.nn.functional as F
from torch import nn

from ksuit.callbacks.online_callbacks import OnlineSegmentationCallback
from ksuit.factory import MasterFactory
from .base import SgdTrainer


class SegmentationTrainer(SgdTrainer):
    def __init__(self, loss_weights=None, ignore_index=-1, **kwargs):
        super().__init__(**kwargs)
        self.loss_weights = loss_weights or {}
        self.ignore_index = ignore_index

    def get_trainer_callbacks(self, model=None):
        # select suited callback_ctor for dataset type (binary/multiclass/multilabel)
        if self.data_container.get_dataset("train").getdim("segmentation") <= 2:
            raise NotImplementedError(f"binary segmentation not supported")
        # create callbacks
        return [
            OnlineSegmentationCallback(
                verbose=False,
                ignore_index=self.ignore_index,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            OnlineSegmentationCallback(
                ignore_index=self.ignore_index,
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def output_shape(self):
        input_shape = self.input_shape
        num_classes = self.data_container.get_dataset("train").getdim("segmentation")
        return num_classes, *input_shape[1:]

    @property
    def dataset_mode(self):
        return "x segmentation"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, batch, reduction="mean"):
            # prepare data
            x = batch["x"].to(self.model.device, non_blocking=True)
            target = batch["segmentation"].to(self.model.device, non_blocking=True)

            # prepare forward
            preds = self.model(x)

            # calculate loss
            if torch.is_tensor(preds):
                preds = dict(main=preds)
            losses = {
                name: F.cross_entropy(pred, target, reduction=reduction, ignore_index=self.trainer.ignore_index)
                for name, pred in preds.items()
            }
            total_loss = 0.
            for name, loss in losses.items():
                if name in self.trainer.loss_weights:
                    total_loss = total_loss + loss * self.trainer.loss_weights[name]
                else:
                    total_loss = total_loss + loss
            losses["total"] = total_loss

            # compose outputs (for callbacks to use)
            outputs = {
                "preds": preds,
                "target": target,
            }
            return losses, outputs
