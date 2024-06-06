from datetime import datetime

import torch.cuda

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.utils.formatting_utils import seconds_to_duration_str


class PeakMemoryCallback(PeriodicCallback):
    def _periodic_callback(self, model, **__):
        if str(model.device) == "cpu":
            return
        self.logger.info(f"max_memory_allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
