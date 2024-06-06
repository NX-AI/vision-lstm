import logging

import torch
from kappadata.samplers import (
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
    InterleavedSampler
)

from ksuit.data.wrappers import ModeWrapper, SubsetWrapper, ShuffleWrapper
from ksuit.distributed import is_distributed
from ksuit.providers import NoopConfigProvider
from ksuit.utils.num_worker_heuristic import get_total_cpu_count, get_fair_cpu_count


class DataContainer:
    def __init__(
            self,
            num_workers=None,
            max_num_workers=None,
            pin_memory=None,
            prefetch_factor=2,
            config_provider=None,
            seed=None,
            **datasets,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.num_workers = num_workers
        self.max_num_workers = max_num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.config_provider = config_provider or NoopConfigProvider()
        seed = seed or int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator().manual_seed(seed)

        self.datasets = datasets
        self.persistent_loaders = {}
        self.added_to_config_provider = False
        # run_type can be adjusted by trainers
        self.run_type = "train"

        # set first dataset as "train" dataset in place of an actual dataset
        if "train" not in self.datasets:
            self.datasets["train"] = list(self.datasets.values())[0]

    def get_dataset(self, key=None, mode=None, max_size=None, shuffle_seed=None, return_type=dict):
        key = key or list(self.datasets.keys())[0]
        dataset = self.datasets[key]
        if shuffle_seed is not None:
            dataset = ShuffleWrapper(dataset=dataset, seed=shuffle_seed)
        if max_size is not None:
            dataset = SubsetWrapper(dataset, end_index=max_size)
        if mode is not None:
            dataset = ModeWrapper(dataset=dataset, mode=mode, return_type=return_type)
        return dataset

    def get_main_sampler(
            self,
            train_dataset,
            num_repeats=1,
            shuffle=True,
    ):
        # TODO port to version that supports instantiation via kind
        if is_distributed():
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())
            self.logger.info(f"main_sampler: DistributedSampler(num_repeats={num_repeats}, shuffle={shuffle})")
            # NOTE: drop_last is required as otherwise len(sampler) can be larger than len(dataset)
            # which results in unconsumed batches from InterleavedSampler
            return DistributedSampler(
                train_dataset,
                num_repeats=num_repeats,
                shuffle=shuffle,
                seed=seed,
                drop_last=True,
            )
        if shuffle:
            self.logger.info(f"main_sampler: RandomSampler(num_repeats={num_repeats})")
            return RandomSampler(train_dataset, num_repeats=num_repeats, generator=self.generator)
        else:
            self.logger.info(f"main_sampler: SequentialSampler")
            return SequentialSampler(train_dataset)

    def get_data_loader(
            self,
            main_sampler,
            main_collator,
            batch_size,
            epochs,
            updates,
            samples,
            configs,
            start_epoch=None,
    ):
        sampler = InterleavedSampler(
            main_sampler=main_sampler,
            batch_size=batch_size,
            configs=configs,
            main_collator=main_collator,
            epochs=epochs,
            updates=updates,
            samples=samples,
            start_epoch=start_epoch,
        )
        if self.num_workers is None:
            num_workers = get_fair_cpu_count()
        else:
            num_workers = self.num_workers
        if self.max_num_workers is not None:
            num_workers = min(self.max_num_workers, num_workers)
        pin_memory = True if self.pin_memory is None else self.pin_memory
        loader = sampler.get_data_loader(
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.prefetch_factor,
        )
        # log properties
        self.logger.info(
            f"created dataloader (batch_size={batch_size} num_workers={loader.num_workers} "
            f"pin_memory={loader.pin_memory} total_cpu_count={get_total_cpu_count()} "
            f"prefetch_factor={loader.prefetch_factor})"
        )
        self.logger.info(f"concatenated dataset properties:")
        for dataset in sampler.dataset.datasets:
            self.logger.info(f"- mode='{dataset.mode}' len={len(dataset)} root_dataset={dataset.root_dataset}")
        # add to wandb config
        if not self.added_to_config_provider:
            self.config_provider.update({
                f"dataloader/num_workers": loader.num_workers,
                f"dataloader/pin_memory": loader.pin_memory,
            })
            self.added_to_config_provider = True
        return loader
