import logging
from functools import partial

from torch.utils.data import Dataset as TorchDataset

from ksuit.data.collators import Collator, ComposeCollator
from ksuit.data.error_messages import getshape_instead_of_getdim
from ksuit.data.errors import UseModeWrapperException
from ksuit.factory import MasterFactory


class Dataset(TorchDataset):
    def __init__(self, collators=None):
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self._collator = MasterFactory.get("collator").create(collators, collate_fn=ComposeCollator)

        # getdim_... is an alias -> should be defined via getshape_...
        getdim_names = [name for name in dir(self) if name.startswith("getdim_")]
        assert len(getdim_names) == 0, getshape_instead_of_getdim(getdim_names)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise UseModeWrapperException

    def __getattr__(self, item):
        # make getdim_... an alias to getshape_...[0]
        if item.startswith("getdim_"):
            return partial(self.getdim, item[len("getdim_"):])
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @property
    def collator(self):
        return self._collator or Collator()

    @collator.setter
    def collator(self, collator):
        self._collator = collator

    @property
    def root_dataset(self):
        return self

    @staticmethod
    def has_wrapper(wrapper):
        return False

    @staticmethod
    def has_wrapper_type(wrapper_type):
        return False

    @property
    def all_wrappers(self):
        return []

    @property
    def all_wrapper_types(self):
        return []

    def get_wrapper_of_type(self, wrapper_type):
        return None

    def get_wrappers_of_type(self, wrapper_type):
        return []

    def set_rng(self, rng):
        return self

    def worker_init_fn(self, *args, **kwargs):
        self.collator.worker_init_fn(*args, **kwargs)

    def getshape(self, kind):
        attr = f"getshape_{kind}"
        assert hasattr(self, attr), f"{type(self).__name__} has no attribute {attr}"
        return getattr(self, attr)()

    def getdim(self, kind):
        shape = self.getshape(kind)
        assert isinstance(shape, tuple) and len(shape) == 1, f"shape {shape} is not 1D"
        return shape[0]

    def __iter__(self):
        """
        torch.utils.data.Dataset doesn't define __iter__
        which makes 'for sample in dataset' run endlessly
        """
        for i in range(len(self)):
            yield self[i]
