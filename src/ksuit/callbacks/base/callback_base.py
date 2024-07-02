import logging
from collections import defaultdict

import kappaprofiler as kp
import torch

from ksuit.distributed import all_gather_nograd
from ksuit.providers.config_providers.base.config_provider_base import ConfigProviderBase
from ksuit.providers.config_providers.noop_config_provider import NoopConfigProvider
from ksuit.providers.path_provider import PathProvider
from ksuit.providers.summary_providers.base.summary_provider_base import SummaryProviderBase
from ksuit.providers.summary_providers.noop_summary_provider import NoopSummaryProvider
from ksuit.utils.data_container import DataContainer
from ksuit.utils.formatting_utils import list_to_string
from ksuit.utils.naming_utils import snake_type_name
from ksuit.utils.update_counter import UpdateCounter
from .writers.checkpoint_writer import CheckpointWriter
from .writers.log_writer import LogWriter


class CallbackBase:
    log_writer_singleton = None

    @property
    def writer(self):
        if CallbackBase.log_writer_singleton is None:
            CallbackBase.log_writer_singleton = LogWriter(
                path_provider=self.path_provider,
                update_counter=self.update_counter,
            )
        return CallbackBase.log_writer_singleton

    @staticmethod
    def flush():
        if CallbackBase.log_writer_singleton is not None:
            CallbackBase.log_writer_singleton.flush()

    @staticmethod
    def finish():
        if CallbackBase.log_writer_singleton is not None:
            CallbackBase.log_writer_singleton.finish()

    def __init__(
            self,
            data_container: DataContainer = None,
            config_provider: ConfigProviderBase = None,
            summary_provider: SummaryProviderBase = None,
            path_provider: PathProvider = None,
            update_counter: UpdateCounter = None,
    ):
        self.data_container = data_container
        self.config_provider = config_provider or NoopConfigProvider()
        self.summary_provider = summary_provider or NoopSummaryProvider()
        self.path_provider = path_provider
        self.update_counter = update_counter

        self.total_data_time = defaultdict(float)
        self.total_forward_time = defaultdict(float)

        # these things are initialized on property access because they require the name/full_name
        # (which can be set from child classes)
        self._callback = None
        # trainer checkpoint requires gathering random states -> all ranks have a checkpoint writer
        self.checkpoint_writer = CheckpointWriter(path_provider=self.path_provider, update_counter=update_counter)

        # check that children only override their implementation methods
        assert type(self).before_training == CallbackBase.before_training
        assert type(self).after_training == CallbackBase.after_training

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

    @property
    def logger(self):
        if self._callback is None:
            self._callback = logging.getLogger(str(self))
        return self._callback

    @torch.no_grad()
    def before_training(self, **kwargs):
        if type(self)._before_training == CallbackBase._before_training:
            return
        with kp.named_profile(f"{self}.before_training"):
            self._before_training(**kwargs)

    @torch.no_grad()
    def after_training(self, **kwargs):
        for dataset_key in self.total_data_time.keys():
            total_data_time = all_gather_nograd(self.total_data_time[dataset_key])
            total_forward_time = all_gather_nograd(self.total_forward_time[dataset_key])
            self.logger.info("------------------")
            self.logger.info(f"{snake_type_name(self)} dataset_key={dataset_key}")
            self.logger.info(f"total_data_time:    {list_to_string(total_data_time)}")
            self.logger.info(f"total_forward_time: {list_to_string(total_forward_time)}")

        if type(self)._after_training == CallbackBase._after_training:
            return
        with kp.named_profile(f"{self}.after_training"):
            self._after_training(**kwargs)

    def _before_training(self, **kwargs):
        pass

    def _after_training(self, **kwargs):
        pass

    def register_root_datasets(self, dataset_config_provider=None, is_mindatarun=False):
        pass

    def resume_from_checkpoint(self, stage_name, stage_id, model):
        pass
