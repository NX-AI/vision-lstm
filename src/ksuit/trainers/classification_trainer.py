import torch
import torch.nn.functional as F
from torch import nn

from ksuit.callbacks.online_callbacks import OnlineAccuracyCallback
from ksuit.factory import MasterFactory
from .base import SgdTrainer


class ClassificationTrainer(SgdTrainer):
    def __init__(self, forward_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.forward_kwargs = MasterFactory.create_dict(forward_kwargs)

    def get_trainer_callbacks(self, model=None):
        # select suited callback_ctor for dataset type (binary/multiclass/multilabel)
        ds = self.data_container.get_dataset("train")
        if ds.getdim("class") <= 2:
            raise NotImplementedError(f"binary classification not supported")
        # create callbacks
        return [
            OnlineAccuracyCallback(
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            OnlineAccuracyCallback(
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def output_shape(self):
        return self.data_container.get_dataset("train").getdim("class"),

    @property
    def dataset_mode(self):
        return "index x class"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, batch, forward_kwargs=None, reduction="mean"):
            # prepare data
            idx = batch["index"]
            x = batch["x"].to(self.model.device, non_blocking=True)
            target = batch["class"].to(self.model.device, non_blocking=True)

            # prepare forward
            if self.model.training:
                assert forward_kwargs is None
                forward_kwargs = self.trainer.forward_kwargs
            else:
                forward_kwargs = forward_kwargs or {}

            # forward
            preds = self.model(x, **forward_kwargs)

            # calculate loss
            if torch.is_tensor(preds):
                preds = dict(main=preds)
            losses = {
                name: F.cross_entropy(preds, target, reduction=reduction)
                for name, preds in preds.items()
            }
            losses["total"] = sum(losses.values())

            # compose outputs (for callbacks to use)
            outputs = {
                "idx": idx,
                "preds": preds,
                "target": target,
            }
            return losses, outputs
