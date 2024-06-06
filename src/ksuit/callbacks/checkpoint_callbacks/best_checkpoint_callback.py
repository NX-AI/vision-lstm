from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.providers import MetricPropertyProvider


class BestCheckpointCallback(PeriodicCallback):
    def __init__(
            self,
            metric_key,
            save_frozen_weights=True,
            tolerances=None,
            model_name=None,
            model_names=None,
            metric_property_provider: MetricPropertyProvider = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric_key = metric_key
        self.model_names = model_names or []
        if model_name is not None:
            self.model_names.append(model_name)
        self.metric_property_provider = metric_property_provider or MetricPropertyProvider()
        self.higher_is_better = self.metric_property_provider.higher_is_better(self.metric_key)
        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")
        self.save_frozen_weights = save_frozen_weights

        # save multiple best models based on tolerance
        self.tolerances_is_exceeded = {tolerance: False for tolerance in tolerances or []}
        self.tolerance_counter = 0
        self.metric_at_exceeded_tolerance = {}

    def state_dict(self):
        return dict(
            best_metric_value=self.best_metric_value,
            tolerances_is_exceeded=self.tolerances_is_exceeded,
            tolerance_counter=self.tolerance_counter,
            metric_at_exceeded_tolerance=self.metric_at_exceeded_tolerance,
        )

    def load_state_dict(self, state_dict):
        self.best_metric_value = state_dict["best_metric_value"]
        self.tolerances_is_exceeded = state_dict["tolerances_is_exceeded"]
        self.tolerance_counter = state_dict["tolerance_counter"]
        self.metric_at_exceeded_tolerance = state_dict["metric_at_exceeded_tolerance"]

    def _before_training(self, **kwargs):
        if len(self.tolerances_is_exceeded) > 0 and self.update_counter.cur_checkpoint.sample > 0:
            raise NotImplementedError(f"{type(self).__name__} with tolerances resuming not implemented")

    def _is_new_best_model(self, metric_value):
        if self.higher_is_better:
            return metric_value > self.best_metric_value
        return metric_value < self.best_metric_value

    # noinspection PyMethodOverriding
    def _periodic_callback(self, trainer, model, **kwargs):
        assert self.metric_key in self.writer.log_cache, (
            f"couldn't find metric_key {self.metric_key} (valid metric keys={list(self.writer.log_cache.keys())}) -> "
            f"make sure the callback that produces the metric_key is called at the same (or higher) frequency and "
            f"is ordered before the {type(self).__name__}"
        )
        metric_value = self.writer.log_cache[self.metric_key]

        if self._is_new_best_model(metric_value):
            # one could also track the model and save it after training
            # this is better in case runs crash or are terminated
            # the runtime overhead is neglegible
            self.logger.info(f"new best model ({self.metric_key}): {self.best_metric_value} --> {metric_value}")
            self.checkpoint_writer.save(
                model=model,
                checkpoint=f"best_model.{self.metric_key.replace('/', '.')}",
                save_optim=False,
                model_names_to_save=self.model_names,
                save_frozen_weights=self.save_frozen_weights,
            )
            self.best_metric_value = metric_value
            self.tolerance_counter = 0
            # log tolerance checkpoints
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded:
                    continue
                self.checkpoint_writer.save(
                    model=model,
                    checkpoint=f"best_model.{self.metric_key.replace('/', '.')}.tolerance{tolerance}",
                    save_optim=False,
                    model_names_to_save=self.model_names,
                )
        else:
            self.tolerance_counter += 1
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded:
                    continue
                if tolerance >= self.tolerance_counter:
                    self.tolerances_is_exceeded[tolerance] = True
                    self.metric_at_exceeded_tolerance[tolerance] = metric_value

    def _after_training(self, **kwargs):
        # best metric doesn't need to be logged as it is summarized anyways
        for tolerance, value in self.metric_at_exceeded_tolerance.items():
            self.logger.info(f"best {self.metric_key} with tolerance={tolerance}: {value}")
