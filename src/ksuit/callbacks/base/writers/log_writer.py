import logging
from collections import defaultdict
from contextlib import contextmanager

import torch
import wandb
import yaml

from ksuit.distributed import is_rank0
from ksuit.providers import PathProvider
from ksuit.utils.update_counter import UpdateCounter


class LogWriter:
    def __init__(self, path_provider: PathProvider, update_counter: UpdateCounter):
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider
        self.update_counter = update_counter
        self.log_entries = []
        self.log_cache = None
        self.is_wandb = wandb.run is not None
        self._postfix = None

    def finish(self):
        if len(self.log_entries) == 0 or not is_rank0():
            return
        entries_uri = self.path_provider.primitive_entries_uri
        self.logger.info(f"writing {len(self.log_entries)} log entries to {entries_uri}")
        # convert into {<key>: {<update0>: <value0>, <update1>: <value1>}}
        result = defaultdict(dict)
        for entry in self.log_entries:
            # update is used instead of wandb's _step
            update = entry["update"]
            for key, value in entry.items():
                if key == "update":
                    continue
                result[key][update] = value
        torch.save(dict(result), entries_uri)
        # yaml is quite inefficient to store large data quantities
        # with open(entries_uri, "w") as f:
        #     yaml.safe_dump(dict(result), f)

    def _log(self, key, value, logger=None, format_str=None):
        if self.log_cache is None:
            self.log_cache = dict(
                epoch=self.update_counter.epoch,
                update=self.update_counter.update,
                sample=self.update_counter.sample,
            )
        if self._postfix is not None:
            key = f"{key}/{self._postfix}"
        assert key not in self.log_cache, f"cant log key '{key}' twice"
        self.log_cache[key] = value
        if logger is not None:
            if format_str is not None:
                value = f"{value:{format_str}}"
            logger.info(f"{key}: {value}")

    def flush(self):
        if self.log_cache is None:
            return
        if self.is_wandb:
            wandb.log(self.log_cache)
        # wandb doesn't support querying offline logfiles so offline mode would have no way to summarize stages
        # also fetching the summaries from the online version potentially takes a long time, occupying GPU servers
        # for primitive tasks
        # -------------------
        # wandb also has weird behavior when lots of logs are done seperately -> collect all log values and log once
        # -------------------
        # check that every log is fully cached (i.e. no update is logged twice)
        if len(self.log_entries) > 0:
            assert self.log_cache["update"] > self.log_entries[-1]["update"]
        # don't keep histograms for primitive logging
        self.log_entries.append({k: v for k, v in self.log_cache.items() if not isinstance(v, wandb.Histogram)})
        self.log_cache = None

    def add_scalar(self, key, value, logger=None, format_str=None):
        if torch.is_tensor(value):
            value = value.item()
        self._log(key, value, logger=logger, format_str=format_str)

    def add_histogram(self, key, data):
        if self.is_wandb:
            self._log(key, wandb.Histogram(data))

    def add_previous_entry(self, entry):
        # only add to wandb as primitive entries are currently based on updates
        # add_previous_entry is only used to copy graphs from other runs into
        # the current run so primitive logging is not needed anyways
        if self.is_wandb:
            wandb.log(entry)

    @contextmanager
    def with_postfix(self, postfix):
        prev_postfix = self._postfix
        if self._postfix is not None:
            self._postfix = f"{self._postfix}/{postfix}"
        else:
            self._postfix = postfix
        yield
        self._postfix = prev_postfix
