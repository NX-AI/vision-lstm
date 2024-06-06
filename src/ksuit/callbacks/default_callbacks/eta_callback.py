import logging
from datetime import datetime, timedelta

import numpy as np

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.distributed import is_rank0
from ksuit.utils.formatting_utils import short_number_str


class EtaCallback(PeriodicCallback):
    class LoggerWasCalledHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.was_called = False

        def emit(self, _):
            self.was_called = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_time = 0.
        self.time_since_last_log = 0.
        self.handler = self.LoggerWasCalledHandler()
        self.epoch_format = None
        self.update_format = None
        self._start_time = None

    def _before_training(self, **kwargs):
        assert is_rank0(), "only use EtaCallback on rank0 process"
        self.epoch_format = f"{int(np.log10(max(1, self.update_counter.end_checkpoint.epoch))) + 1}d"
        self.update_format = f"{int(np.log10(self.update_counter.end_checkpoint.update)) + 1}d"
        self.every_n_epochs_format = f"{int(np.log10(self.every_n_epochs)) + 1}d" if self.every_n_epochs else None
        self.every_n_updates_format = f"{int(np.log10(self.every_n_updates)) + 1}d" if self.every_n_updates else None

        if self.every_n_epochs:
            self.updates_per_log_interval_format = f"{int(np.log10(self.update_counter.updates_per_epoch)) + 1}d"
        elif self.every_n_updates:
            self.updates_per_log_interval_format = self.every_n_updates_format
        elif self.every_n_samples:
            self.updates_per_every_n_samples = np.ceil(self.every_n_samples / self.update_counter.effective_batch_size)
            self.updates_per_log_interval_format = f"{int(np.log10(self.updates_per_every_n_samples)) + 1}d"
        else:
            self.updates_per_log_interval_format = None
        self._start_time = datetime.now()

    def _track_after_update_step(self, times, **kwargs):
        cur_epoch = self.update_counter.cur_checkpoint.epoch - self.update_counter.start_checkpoint.epoch
        cur_update = self.update_counter.cur_checkpoint.update - self.update_counter.start_checkpoint.update
        cur_sample = self.update_counter.cur_checkpoint.sample - self.update_counter.start_checkpoint.sample
        now = datetime.now()
        # reset time_since_last_log on new log interval
        if self.should_log_after_epoch(self.update_counter.cur_checkpoint) and self.update_counter.is_full_epoch:
            self.time_since_last_log = 0.
        if self.should_log_after_update(self.update_counter.cur_checkpoint):
            self.time_since_last_log = 0.
        if self.should_log_after_sample(self.update_counter.cur_checkpoint, self.update_counter.effective_batch_size):
            self.time_since_last_log = 0.

        if self.every_n_epochs:
            last_epoch = self.every_n_epochs * (cur_epoch // self.every_n_epochs)
            updates_at_last_log = last_epoch * self.update_counter.updates_per_epoch
            updates_since_last_log = cur_update - updates_at_last_log
            updates_per_log_interval = self.every_n_epochs * self.update_counter.updates_per_epoch
            if updates_since_last_log == 0:
                updates_since_last_log = updates_per_log_interval
        elif self.every_n_updates:
            updates_since_last_log = cur_update % self.every_n_updates
            updates_per_log_interval = self.every_n_updates
        elif self.every_n_samples:
            samples_since_last_log = cur_sample % self.every_n_samples
            samples_at_last_log = cur_sample - samples_since_last_log
            updates_at_last_log = samples_at_last_log // self.update_counter.effective_batch_size
            superflous_samples_at_last_log = samples_at_last_log % self.update_counter.effective_batch_size
            updates_since_last_log = cur_update - updates_at_last_log
            samples_for_cur_log_interval = self.every_n_samples - superflous_samples_at_last_log
            updates_per_log_interval = int(
                np.ceil(samples_for_cur_log_interval / self.update_counter.effective_batch_size)
            )
        else:
            updates_since_last_log = None
            updates_per_log_interval = None

        # add time
        time_increment = times["data_time"] + times["update_time"]
        self.total_time += time_increment
        self.time_since_last_log += time_increment
        average_update_time = self.total_time / cur_update

        # training ETA
        progress = (
                (self.update_counter.cur_checkpoint.update - self.update_counter.start_checkpoint.update) /
                (self.update_counter.end_checkpoint.update - self.update_counter.start_checkpoint.update)
        )
        past_training_time = now - self._start_time
        estimated_duration = past_training_time / progress
        training_eta = self._start_time + estimated_duration
        remaining_training_time = training_eta - now

        logstr = (
            f"E {format(cur_epoch, self.epoch_format)}/{self.update_counter.end_checkpoint.epoch} "
            f"U {format(cur_update, self.update_format)}/{self.update_counter.end_checkpoint.update} "
            f"S {short_number_str(cur_sample):>6}/"
            f"{short_number_str(self.update_counter.end_checkpoint.sample)} | "
        )
        # log interval ETA
        if self.updates_per_log_interval_format is not None:
            updates_till_next_log = updates_per_log_interval - updates_since_last_log
            time_till_next_log = timedelta(seconds=updates_till_next_log * average_update_time)
            next_log_eta = now + time_till_next_log
            # convert to datetime for formatting
            past_next_log_time = datetime.utcfromtimestamp(self.time_since_last_log)
            time_till_next_log = datetime.utcfromtimestamp(time_till_next_log.total_seconds())
            logstr += (
                f"next_log {format(updates_since_last_log, self.updates_per_log_interval_format)}/"
                f"{format(updates_per_log_interval, self.updates_per_log_interval_format)} | "
                f"next_log_eta {next_log_eta.strftime('%H:%M:%S')} "
                f"({time_till_next_log.strftime('%M:%S')}->{past_next_log_time.strftime('%M:%S')}) | "
            )
        logstr += (
            # f"training_eta {training_eta.strftime('%d-%H:%M:%S')} "
            # f"({seconds_to_duration_str(remaining_training_time.total_seconds())}->"
            # f"{seconds_to_duration_str(past_training_time.total_seconds())}) | "
            f"avg_update {average_update_time:.2f}s"
        )
        if self.handler.was_called:
            print(logstr)
            self.handler.was_called = False
        else:
            print(logstr, end="\r")

    def _periodic_callback(self, **_):
        print()

    def _after_training(self, **_):
        logging.getLogger().removeHandler(self.handler)
