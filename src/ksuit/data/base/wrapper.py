import inspect
from functools import partial

from .dataset import Dataset


class Wrapper(Dataset):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item):
        if item.startswith("getitem_"):
            # ctx is optional for getitem methods but wrappers should propagate it
            func = getattr(self.dataset, item)
            return partial(self._call_getitem, func)
        if item == "dataset":
            return getattr(super(), item)
        # make getdim_... an alias to getshape_...[0]
        if item.startswith("getdim_"):
            return partial(self.getdim, item[len("getdim_"):])
        return getattr(self.dataset, item)

    def _call_getitem(self, func, idx, ctx=None, *args, **kwargs):
        root_func = func
        while isinstance(root_func, partial):
            root_func = root_func.args[0]
        if "ctx" in inspect.getfullargspec(root_func).args:
            return func(idx, *args, **kwargs, ctx=ctx)
        else:
            return func(idx, *args, **kwargs)

    @property
    def collator(self):
        assert self._collator is None, "register collator on root datset"
        return self.dataset.collator

    @collator.setter
    def collator(self, collator):
        raise RuntimeError("register collator on root datset")

    @property
    def root_dataset(self):
        # Dataset implements root_dataset -> __getattr__ doesn't trigger
        return self.dataset.root_dataset

    def has_wrapper(self, wrapper):
        if self == wrapper:
            return True
        return self.dataset.has_wrapper(wrapper)

    def has_wrapper_type(self, wrapper_type):
        if type(self) == wrapper_type:
            return True
        return self.dataset.has_wrapper_type(wrapper_type)

    @property
    def all_wrappers(self):
        return [self] + self.dataset.all_wrappers

    @property
    def all_wrapper_types(self):
        return [type(self)] + self.dataset.all_wrapper_types

    def get_wrapper_of_type(self, wrapper_type):
        wrappers = self.get_wrappers_of_type(wrapper_type)
        if len(wrappers) == 0:
            return None
        assert len(wrappers) == 1
        return wrappers[0]

    def get_wrappers_of_type(self, wrapper_type):
        wrappers = self.dataset.get_wrappers_of_type(wrapper_type)
        if type(self) == wrapper_type:
            return [self] + wrappers
        return wrappers

    def set_rng(self, rng):
        super().set_rng(rng)
        self.dataset.set_rng(rng)
        return self

    def worker_init_fn(self, rank, **kwargs):
        super().worker_init_fn(rank=rank, **kwargs)
        self.dataset.worker_init_fn(rank, **kwargs)
