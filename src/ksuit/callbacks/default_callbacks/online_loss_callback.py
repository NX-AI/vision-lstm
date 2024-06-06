from collections import defaultdict

import torch

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.distributed import all_reduce_mean_nograd, all_gather_nograd


class OnlineLossCallback(PeriodicCallback):
    def __init__(self, verbose, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.tracked_losses = defaultdict(list)

    def _track_after_accumulation_step(self, losses, **kwargs):
        for name, loss in losses.items():
            self.tracked_losses[name].append(loss.detach())

    def _periodic_callback(self, trainer, **_):
        for name, tracked_loss in self.tracked_losses.items():
            mean_loss = all_reduce_mean_nograd(torch.stack(tracked_loss).mean())
            if not trainer.skip_nan_loss and torch.isnan(mean_loss):
                losses = all_gather_nograd(torch.stack(tracked_loss))
                num_nans = torch.isnan(losses).sum()
                msg = f"encountered nan loss ({num_nans.item()} nans): {losses}"
                self.logger.error(msg)
                raise RuntimeError(msg)
            self.writer.add_scalar(
                key=f"loss/online/{name}/{self.to_short_interval_string()}",
                value=mean_loss,
                logger=self.logger if self.verbose else None,
            )
        self.tracked_losses.clear()
