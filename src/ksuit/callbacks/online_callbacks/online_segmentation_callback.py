from collections import defaultdict

import torch
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_jaccard_index

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.distributed import all_reduce_mean_nograd


class OnlineSegmentationCallback(PeriodicCallback):
    def __init__(self, verbose=True, ignore_index=-1, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.ignore_index = ignore_index
        self.tracked_accs = defaultdict(list)
        self.target_dim = None

    def _before_training(self, model, **kwargs):
        self.target_dim = self.data_container.get_dataset("train").getdim("segmentation")

    def _track_after_accumulation_step(self, update_outputs, **kwargs):
        target = update_outputs["target"]
        # convert back to long (e.g. when label smoothing is used)
        if target.dtype != torch.long:
            target = target.argmax(dim=1)

        for name, pred in update_outputs["preds"].items():
            accuracy = multiclass_accuracy(
                preds=pred,
                target=target,
                top_k=1,
                num_classes=self.target_dim,
                average="micro",
                ignore_index=self.ignore_index,
            ).item()
            self.tracked_accs[name].append(accuracy)

    def _periodic_callback(self, **_):
        kwargs = dict(logger=self.logger, format_str=".6f") if self.verbose else {}
        for name, value in self.tracked_accs.items():
            mean = all_reduce_mean_nograd(torch.tensor(value).mean())
            self.writer.add_scalar(
                key=f"accuracy1/online/micro/{name}/{self.to_short_interval_string()}",
                value=mean,
                **kwargs,
            )
        self.tracked_accs.clear()
