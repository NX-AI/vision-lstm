import math

import kappaprofiler as kp
import numpy as np
import torch
from kappadata.samplers import InterleavedSamplerConfig
from torch.utils.data import SequentialSampler, DistributedSampler
from tqdm import tqdm

from ksuit.data.wrappers import ModeWrapper
from ksuit.distributed import is_distributed, is_managed, is_rank0, all_gather_nograd_clipped
from ksuit.utils.naming_utils import snake_type_name
from ksuit.utils.noop_tqdm import NoopTqdm
from .callback_base import CallbackBase


class PeriodicCallback(CallbackBase):
    def __init__(
            self,
            every_n_epochs: int = None,
            every_n_updates: int = None,
            every_n_samples: int = None,
            batch_size: int = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.every_n_epochs = every_n_epochs
        self.every_n_updates = every_n_updates
        self.every_n_samples = every_n_samples
        self.batch_size = batch_size
        self._sampler_configs = {}
        self._sampler_config_names = {}
        # whenever a callback wants to iterate over a dataset -> check if it does so in the correct order
        # (order of iteration needs to be the same order as registration)
        self.__sampler_configs_counter = 0

        # if stuff is tracked -> multiple interval types lead to inconsistent results
        if (
                type(self)._track_after_accumulation_step != PeriodicCallback._track_after_accumulation_step or
                type(self)._track_after_update_step != PeriodicCallback._track_after_update_step
        ):
            assert sum([
                self.every_n_epochs is not None,
                self.every_n_updates is not None,
                self.every_n_samples is not None,
            ]) <= 1, "tracking callbacks can't have multiple interval types"

        # periodic callback requires update_counter
        assert self.update_counter is not None

        # check that children only override their implementation methods
        assert type(self).track_after_accumulation_step == PeriodicCallback.track_after_accumulation_step
        assert type(self).track_after_update_step == PeriodicCallback.track_after_update_step
        assert type(self).after_update == PeriodicCallback.after_update
        assert type(self).after_epoch == PeriodicCallback.after_epoch
        assert type(self).register_sampler_configs == PeriodicCallback.register_sampler_configs
        # this might be confused with register_sampler_configs method and accidentally overwritten
        assert type(self)._register_sampler_config_from_key == PeriodicCallback._register_sampler_config_from_key

    def __str__(self):
        detail_str = self._to_string() or ""
        return f"{type(self).__name__}({self.get_interval_string_verbose()}{detail_str})"

    def _to_string(self):
        return None

    def _register_sampler_config_from_key(self, key, mode, max_size=None):
        dataset = self.data_container.get_dataset(key=key, mode=mode, max_size=max_size)
        return self.__register_sampler_config(dataset=dataset, mode=mode, name=key, collator=dataset.collator)

    def _register_sampler_config_from_dataset(self, dataset, mode, name):
        assert not isinstance(dataset, ModeWrapper)
        dataset = ModeWrapper(dataset=dataset, mode=mode, return_ctx=True)
        return self.__register_sampler_config(dataset=dataset, mode=mode, name=name)

    def __register_sampler_config(self, dataset, mode, name, collator=None):
        assert len(dataset) > 0
        config = InterleavedSamplerConfig(
            sampler=DistributedSampler(dataset, shuffle=False) if is_distributed() else SequentialSampler(dataset),
            every_n_epochs=self.every_n_epochs,
            every_n_updates=self.every_n_updates,
            every_n_samples=self.every_n_samples,
            collator=collator,
            batch_size=self.batch_size,
        )
        config_id = len(self._sampler_configs)
        self._sampler_configs[config_id] = config
        self._sampler_config_names[config_id] = f"{name}.{mode.replace(' ', '.')}"
        return config_id

    def register_sampler_configs(self, trainer):
        assert len(self._sampler_configs) == 0
        self._register_sampler_configs(trainer)
        return self._sampler_configs.values(), self._sampler_config_names.values()

    def _register_sampler_configs(self, trainer):
        pass

    def should_log_after_epoch(self, checkpoint):
        if self.every_n_epochs is not None:
            return checkpoint.epoch % self.every_n_epochs == 0
        return False

    def should_log_after_update(self, checkpoint):
        if self.every_n_updates is not None:
            return checkpoint.update % self.every_n_updates == 0
        return False

    def should_log_after_sample(self, checkpoint, effective_batch_size):
        if self.every_n_samples is not None:
            last_update_samples = checkpoint.sample - effective_batch_size
            prev_log_step = int(last_update_samples / self.every_n_samples)
            cur_log_step = int(checkpoint.sample / self.every_n_samples)
            if cur_log_step > prev_log_step:
                return True
        return False

    def before_every_update(self, **kwargs):
        pass

    def before_every_backward(self, **kwargs):
        pass

    def before_every_accumulation_step(self, **kwargs):
        pass

    def _track_after_accumulation_step(self, **kwargs):
        pass

    def _track_after_update_step(self, **kwargs):
        pass

    def _periodic_callback(self, interval_type, **kwargs):
        pass

    @torch.no_grad()
    def track_after_accumulation_step(self, **kwargs):
        if type(self)._track_after_accumulation_step == PeriodicCallback._track_after_accumulation_step:
            return
        with kp.named_profile(f"{self}.track_after_accumulation_step"):
            self._track_after_accumulation_step(**kwargs)

    @torch.no_grad()
    def track_after_update_step(self, **kwargs):
        if type(self)._track_after_update_step == PeriodicCallback._track_after_update_step:
            return
        with kp.named_profile(f"{self}.track_after_update_step"):
            self._track_after_update_step(**kwargs)

    @torch.no_grad()
    def after_epoch(self, **kwargs):
        if type(self)._periodic_callback == PeriodicCallback._periodic_callback:
            return
        if self.should_log_after_epoch(self.update_counter.cur_checkpoint):
            with kp.named_profile(f"{self}.after_epoch"):
                self._periodic_callback(interval_type="epoch", **kwargs)

    @torch.no_grad()
    def after_update(self, effective_batch_size, **kwargs):
        if type(self)._periodic_callback == PeriodicCallback._periodic_callback:
            return
        if self.should_log_after_update(self.update_counter.cur_checkpoint):
            with kp.named_profile(f"{self}.after_update"):
                self._periodic_callback(interval_type="update", **kwargs)
        if self.should_log_after_sample(self.update_counter.cur_checkpoint, effective_batch_size):
            with kp.named_profile(f"{self}.after_sample"):
                self._periodic_callback(interval_type="sample", **kwargs)

    @property
    def updates_till_next_log(self):
        updates_per_log_interval = self.updates_per_log_interval
        return updates_per_log_interval - self.update_counter.cur_checkpoint.update % updates_per_log_interval

    @property
    def updates_per_log_interval(self):
        if self.every_n_epochs is not None:
            assert self.every_n_updates is None and self.every_n_samples is None
            return self.update_counter.updates_per_epoch * self.every_n_epochs
        if self.every_n_updates is not None:
            assert self.every_n_epochs is None and self.every_n_samples is None
            return self.every_n_updates
        if self.every_n_samples is not None:
            assert self.every_n_epochs is None and self.every_n_updates is None
            # NOTE: uneven every_n_samples not supported
            assert self.every_n_samples % self.update_counter.effective_batch_size == 0
            return int(self.every_n_samples / self.update_counter.effective_batch_size)
        raise RuntimeError

    def get_interval_string_verbose(self):
        results = []
        if self.every_n_epochs is not None:
            results.append(f"every_n_epochs={self.every_n_epochs}")
        if self.every_n_updates is not None:
            results.append(f"every_n_updates={self.every_n_updates}")
        if self.every_n_samples is not None:
            results.append(f"every_n_samples={self.every_n_samples}")
        return ",".join(results)

    def to_short_interval_string(self):
        results = []
        if self.every_n_epochs is not None:
            results.append(f"E{self.every_n_epochs}")
        if self.every_n_updates is not None:
            results.append(f"U{self.every_n_updates}")
        if self.every_n_samples is not None:
            results.append(f"S{self.every_n_samples}")
        return "_".join(results)

    def iterate_over_dataset(
            self,
            forward_fn,
            config_id,
            batch_size,
            data_iter=None,
            use_collate_fn=True,
    ):
        assert config_id == self.__sampler_configs_counter
        config = self._sampler_configs[self.__sampler_configs_counter]
        dataset_name = self._sampler_config_names[self.__sampler_configs_counter]
        if isinstance(config.sampler, DistributedSampler):
            global_dataset_len = len(config.sampler.dataset)
        else:
            global_dataset_len = len(config.sampler)
        local_dataset_len = len(config.sampler)
        num_batches = math.ceil(local_dataset_len / (config.batch_size or batch_size))
        self.__sampler_configs_counter = (self.__sampler_configs_counter + 1) % len(self._sampler_configs)

        # iterate
        data_times = []
        forward_times = []
        forward_results = []
        pbar_ctor = NoopTqdm if is_managed() or not is_rank0() else tqdm
        for _ in pbar_ctor(iterable=range(num_batches)):
            # load data
            with kp.Stopwatch() as data_sw:
                batch = next(data_iter)
            data_times.append(data_sw.elapsed_seconds)
            # forward
            with kp.Stopwatch() as forward_sw:
                forward_result = forward_fn(batch)
            forward_times.append(forward_sw.elapsed_seconds)
            forward_results.append(forward_result)

        # profiling book keeping
        mean_data_time = float(np.mean(data_times))
        mean_forward_time = float(np.mean(forward_times))
        prefix = f"profiling/{snake_type_name(self)}/{dataset_name}"
        self.logger.info(f"{prefix}: data={mean_data_time:.2f} forward={mean_forward_time:.2f}")
        # NOTE: removed because it bloats wandb
        # if self.update_counter is not None:
        #     self.writer.add_scalar(f"{prefix}/data_time", mean_data_time)
        #     self.writer.add_scalar(f"{prefix}/forward_times", mean_forward_time)
        self.total_data_time[dataset_name] += mean_data_time
        self.total_forward_time[dataset_name] += mean_forward_time

        # collate
        if use_collate_fn:
            single_output = False
            if not isinstance(forward_results[0], tuple):
                forward_results = [(fwr,) for fwr in forward_results]
                single_output = True
            collated = [
                self._collate_result(result, global_dataset_len=global_dataset_len)
                for result in zip(*forward_results)
            ]

            if single_output:
                return collated[0]
        else:
            collated = forward_results

        return collated

    @staticmethod
    def _collate_tensors(tensors):
        if tensors[0].ndim == 0:
            return torch.stack(tensors)
        return torch.concat(tensors)

    @staticmethod
    def _collate_result(result, global_dataset_len):
        if isinstance(result[0], dict):
            # tuple[dict] -> dict[tensor]
            result = {k: PeriodicCallback._collate_tensors([r[k] for r in result]) for k in result[0].keys()}
            # gather
            result = {k: all_gather_nograd_clipped(v, global_dataset_len) for k, v in result.items()}
        else:
            if isinstance(result[0], list):
                # List[List[Tensor]] -> List[Tensor]
                result = [torch.concat(item) for item in zip(*result)]
                result = [all_gather_nograd_clipped(item, global_dataset_len) for item in result]
            elif result[0] is None:
                return None
            else:
                if torch.is_tensor(result[0]):
                    # List[Tensor] -> Tensor
                    if result[0].ndim == 0:
                        result = torch.stack(result)
                    else:
                        result = torch.concat(result)
                else:
                    result = torch.tensor(result)
                result = all_gather_nograd_clipped(result, global_dataset_len)
        return result
