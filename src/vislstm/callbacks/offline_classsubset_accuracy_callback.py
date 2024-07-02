from functools import partial

from torchmetrics.functional.classification import multiclass_accuracy

from ksuit.callbacks.base.periodic_callback import PeriodicCallback


class OfflineClassSubsetAccuracyCallback(PeriodicCallback):
    def __init__(self, dataset_key, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.__config_id = None
        self.class_subset_indices = None
        self.num_classes = None

    def _before_training(self, model, **kwargs):
        dataset = self.data_container.get_dataset(self.dataset_key)
        self.class_subset_indices = dataset.class_subset_indices
        self.num_classes = dataset.subset_num_classes

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode="x class")

    def _forward(self, batch, model, trainer):
        x = batch["x"]
        cls = batch["class"]
        x = x.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            predictions = model.classify(x)
        # only use logits for the actual available classes
        predictions = {
            name: prediction[:, self.class_subset_indices]
            for name, prediction in predictions.items()
        }
        return predictions, cls.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        predictions, target = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # log
        target = target.to(model.device, non_blocking=True)
        for prediction_name, y_hat in predictions.items():
            # accuracy
            acc = multiclass_accuracy(
                preds=y_hat,
                target=target,
                num_classes=self.num_classes,
                average="micro",
            )
            acc_key = f"accuracy1/{self.dataset_key}/{prediction_name}"
            self.writer.add_scalar(acc_key, acc, logger=self.logger, format_str=".6f")
