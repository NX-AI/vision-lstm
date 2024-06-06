import numpy as np

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.distributed import all_gather_nograd
from ksuit.utils.formatting_utils import list_to_string


class TrainTimeCallback(PeriodicCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_data_times = []
        self.update_times = []
        self.total_train_data_time = 0.
        self.total_update_time = 0.

    def _track_after_update_step(self, **kwargs):
        times = kwargs["times"]
        self.train_data_times.append(times["data_time"])
        self.update_times.append(times["update_time"])

    def _periodic_callback(self, interval_type, **_):
        sum_data_time = np.sum(self.train_data_times)
        sum_update_time = np.sum(self.update_times)
        mean_data_time = sum_data_time / len(self.train_data_times)
        mean_update_time = sum_update_time / len(self.update_times)
        self.total_train_data_time += sum_data_time
        self.total_update_time += sum_update_time
        self.train_data_times.clear()
        self.update_times.clear()

        # gather for all devices
        mean_data_times = all_gather_nograd(mean_data_time)
        mean_update_times = all_gather_nograd(mean_update_time)

        # removed because it bloats wandb metrics/summary
        # for i, (mean_data_time, mean_update_time) in enumerate(zip(mean_data_times, mean_update_times)):
        #     # ideally this would have a key like system/<key> but wandb doesn't like that
        #     self.writer.add_scalar(f"profiling/train_data_time/{i}/{interval_type}", mean_data_time)
        #     self.writer.add_scalar(f"profiling/train_update_time/{i}/{interval_type}", mean_update_time)

        self.logger.info(f"data={list_to_string(mean_data_times)} update={list_to_string(mean_update_times)}")

    def _after_training(self, **_):
        total_data_time = all_gather_nograd(self.total_train_data_time)
        total_update_time = all_gather_nograd(self.total_update_time)
        self.logger.info("------------------")
        self.logger.info(f"total_train_data_time:   {list_to_string(total_data_time)}")
        self.logger.info(f"total_update_time: {list_to_string(total_update_time)}")
