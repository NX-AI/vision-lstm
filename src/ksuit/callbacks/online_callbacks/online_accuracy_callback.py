from collections import defaultdict

import torch
from torchmetrics.functional.classification import multiclass_accuracy

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.distributed import all_reduce_mean_grad


class OnlineAccuracyCallback(PeriodicCallback):
    def __init__(self, verbose=True, topk=None, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.tracked_accs = defaultdict(lambda: defaultdict(list))
        self.topk = topk or [1]
        self.target_dim = None

    def _before_training(self, model, **kwargs):
        assert len(model.output_shape) == 1
        self.target_dim = self.data_container.get_dataset("train").getdim("class")

    def _track_after_accumulation_step(self, update_outputs, **kwargs):
        target = update_outputs["target"]
        # convert back to long (e.g. when label smoothing is used)
        if target.dtype != torch.long:
            target = target.argmax(dim=1)

        for name, preds_value in update_outputs["preds"].items():
            for topk in self.topk:
                acc = multiclass_accuracy(
                    preds=preds_value,
                    target=target,
                    top_k=topk,
                    num_classes=self.target_dim,
                    average="micro",
                )
                self.tracked_accs[name][topk].append(acc)

    def _periodic_callback(self, **_):
        kwargs = dict(logger=self.logger, format_str=".6f") if self.verbose else {}
        for name, tracked_prediction in self.tracked_accs.items():
            for topk, tracked_acc in tracked_prediction.items():
                mean_acc = all_reduce_mean_grad(torch.stack(tracked_acc).mean())
                self.writer.add_scalar(
                    key=f"accuracy{topk}/online/{name}/{self.to_short_interval_string()}",
                    value=mean_acc,
                    **kwargs,
                )
        self.tracked_accs.clear()
