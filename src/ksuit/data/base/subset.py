import inspect
from functools import partial

from .wrapper import Wrapper


class Subset(Wrapper):
    def __init__(self, dataset, indices, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, item):
        if item.startswith("getitem_"):
            # all methods starting with getitem_ are called with self.indices[idx]
            # ctx is optional for getitem methods but wrappers should propagate it
            func = getattr(self.dataset, item)
            return partial(self._call_getitem, func)
        if item == "dataset":
            return getattr(super(), item)
        if item.startswith("getall_"):
            # subsample getitem_ with the indices
            return partial(self._call_getall, item)
        return getattr(self.dataset, item)

    def _call_getitem(self, func, idx, ctx=None, *args, **kwargs):
        root_func = func
        while isinstance(root_func, partial):
            root_func = root_func.args[0]
        if "ctx" in inspect.getfullargspec(root_func).args:
            return func(int(self.indices[idx]), *args, **kwargs, ctx=ctx)
        else:
            return func(int(self.indices[idx]), *args, **kwargs)

    def _call_getall(self, item):
        result = getattr(self.dataset, item)()
        return [result[int(i)] for i in self.indices]

    def __iter__(self):
        for i in range(len(self.indices)):
            yield self[self.indices[int(i)]]
