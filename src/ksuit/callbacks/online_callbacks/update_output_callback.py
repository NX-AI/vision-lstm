from collections import defaultdict

import numpy as np
import torch

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.distributed import all_reduce_mean_grad, all_gather_nograd


class UpdateOutputCallback(PeriodicCallback):
    def __init__(
            self,
            keys=None,
            patterns=None,
            verbose=False,
            reduce="mean",
            log_output=True,
            save_output=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert keys is None or (isinstance(keys, list) and all(isinstance(k, str) for k in keys))
        assert patterns is None or (isinstance(patterns, list) and all(isinstance(p, str) for p in patterns))
        self.patterns = patterns or []
        self.keys = keys or []
        assert len(self.keys) > 0 or len(self.patterns) > 0
        self.verbose = verbose
        self.tracked_values = defaultdict(list)
        assert reduce in ["mean", "last"]
        self.reduce = reduce
        self.log_output = log_output
        self.save_output = save_output
        if save_output:
            self.out = self.path_provider.stage_output_path / "update_outputs"
            self.out.mkdir(exist_ok=True)
        else:
            self.out = None

    def _to_string(self):
        return f", keys={self.keys}, patterns={self.patterns}"

    def _track_after_accumulation_step(self, update_outputs, **kwargs):
        if self.reduce == "last" and self.updates_till_next_log > 1:
            return
        if len(self.keys) > 0:
            for key in self.keys:
                value = update_outputs[key]
                if torch.is_tensor(value):
                    value = value.detach()
                self.tracked_values[key].append(value)
        if len(self.patterns) > 0:
            for key, value in update_outputs.items():
                for pattern in self.patterns:
                    if pattern in key:
                        value = update_outputs[key]
                        if torch.is_tensor(value):
                            value = value.detach()
                        self.tracked_values[key].append(value)

    def _periodic_callback(self, **_):
        for key, tracked_values in self.tracked_values.items():
            if self.reduce == "mean":
                if torch.is_tensor(tracked_values[0]):
                    reduced_value = torch.stack(tracked_values).mean()
                else:
                    reduced_value = float(np.mean(tracked_values))
                reduced_value = all_reduce_mean_grad(reduced_value)
            elif self.reduce == "last":
                assert len(tracked_values) == 1
                reduced_value = all_gather_nograd(tracked_values[0])
            else:
                raise NotImplementedError
            if self.log_output:
                assert reduced_value.numel() == 1
                self.writer.add_scalar(
                    key=f"{key}/{self.to_short_interval_string()}",
                    value=reduced_value,
                    logger=self.logger if self.verbose else None,
                    format_str=".5f",
                )
            if self.save_output:
                uri = self.out / f"{key}_{self.to_short_interval_string()}_{self.update_counter.cur_checkpoint}.th"
                torch.save(reduced_value, uri)
        self.tracked_values.clear()
