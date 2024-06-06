import logging

from ksuit.utils.param_checking import check_exclusive


class EarlyStopperBase:
    def __init__(
            self,
            every_n_epochs=None,
            every_n_updates=None,
            every_n_samples=None,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        assert check_exclusive(every_n_epochs, every_n_updates, every_n_samples), \
            "specify only one of every_n_epochs/every_n_updates/every_n_samples"
        self.every_n_epochs = every_n_epochs
        self.every_n_updates = every_n_updates
        self.every_n_samples = every_n_samples

    def to_short_interval_string(self):
        results = []
        if self.every_n_epochs is not None:
            results.append(f"E{self.every_n_epochs}")
        if self.every_n_updates is not None:
            results.append(f"U{self.every_n_updates}")
        if self.every_n_samples is not None:
            results.append(f"S{self.every_n_samples}")
        return "_".join(results)

    def should_stop_after_sample(self, checkpoint, effective_batch_size):
        if self.every_n_samples is not None:
            last_update_samples = checkpoint.sample - effective_batch_size
            prev_log_step = int(last_update_samples / self.every_n_samples)
            cur_log_step = int(checkpoint.sample / self.every_n_samples)
            if cur_log_step > prev_log_step:
                return self._should_stop()
        return False

    def should_stop_after_update(self, checkpoint):
        if self.every_n_updates is None or checkpoint.update % self.every_n_updates != 0:
            return False
        return self._should_stop()

    def should_stop_after_epoch(self, checkpoint):
        if self.every_n_epochs is None or checkpoint.epoch % self.every_n_epochs != 0:
            return False
        return self._should_stop()

    def _should_stop(self):
        raise NotImplementedError
