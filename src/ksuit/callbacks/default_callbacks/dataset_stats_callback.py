from ksuit.callbacks.base.callback_base import CallbackBase


class DatasetStatsCallback(CallbackBase):
    def _before_training(self, **_):
        for dataset_key, dataset in self.data_container.datasets.items():
            self.summary_provider[f"ds_stats/{dataset_key}/len"] = len(dataset)
            self.logger.info(f"{dataset_key}: {len(dataset)} samples")
