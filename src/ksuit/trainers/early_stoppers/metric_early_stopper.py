from ksuit.callbacks import CallbackBase
from .base.early_stopper_base import EarlyStopperBase
from ksuit.providers import MetricPropertyProvider

class MetricEarlyStopper(EarlyStopperBase):
    def __init__(self, metric_key, tolerance, metric_property_provider: MetricPropertyProvider = None, **kwargs):
        super().__init__(**kwargs)
        self.metric_key = metric_key
        self.metric_property_provider = metric_property_provider or MetricPropertyProvider()
        self.higher_is_better = self.metric_property_provider.higher_is_better(metric_key)
        assert tolerance is not None and tolerance >= 1, "tolerance has to be >= 1"
        self.tolerance = tolerance
        self.tolerance_counter = 0
        self.best_metric = -float("inf") if self.higher_is_better else float("inf")

    def _metric_improved(self, cur_metric):
        if self.higher_is_better:
            return cur_metric > self.best_metric
        return cur_metric < self.best_metric

    def _should_stop(self):
        writer = CallbackBase.log_writer_singleton
        assert writer is not None
        assert self.metric_key in writer.log_cache, (
            f"couldn't find metric_key {self.metric_key} (valid metric_keys={writer.log_cache.keys()}) -> "
            "make sure every_n_epochs/every_n_updates/every_n_samples is aligned with the corresponding callback"
        )
        cur_metric = writer.log_cache[self.metric_key]

        if self._metric_improved(cur_metric):
            self.logger.info(f"{self.metric_key} improved: {self.best_metric} --> {cur_metric}")
            self.best_metric = cur_metric
            self.tolerance_counter = 0
        else:
            self.tolerance_counter += 1
            cmp_str = "<=" if self.higher_is_better else ">="
            stop_training_str = " --> stop training" if self.tolerance_counter >= self.tolerance else ""
            self.logger.info(
                f"{self.metric_key} stagnated: {self.best_metric} {cmp_str} {cur_metric} "
                f"({self.tolerance_counter}/{self.tolerance}){stop_training_str}"
            )

        return self.tolerance_counter >= self.tolerance
