import inspect
from functools import partial

from ksuit.data import Wrapper, Dataset


class ModeWrapper(Wrapper):
    def __init__(self, dataset: Dataset, mode: str, return_type=tuple, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        assert return_type in [dict, tuple]
        self.mode = mode
        self.return_type = return_type

        # split mode into _getitem_fns
        self._getitem_fns = {}
        for item in mode.split(" "):
            if item == "index":
                self._getitem_fns[item] = self._getitem_index
            elif item.startswith("ctx."):
                ctx_key = item[len("ctx."):]
                self._getitem_fns[item] = partial(self._getitem_from_ctx, ctx_key=ctx_key)
            else:
                fn_name = f"getitem_{item}"
                # check that dataset implements getitem (wrappers can use the getitem of their child)
                assert hasattr(self.dataset, fn_name), f"{type(self.dataset)} has no method getitem_{item}"
                self._getitem_fns[item] = getattr(self.dataset, fn_name)

    @staticmethod
    def has_item(mode, item):
        return item in mode.split(" ")

    @staticmethod
    def add_item(mode, item):
        if ModeWrapper.has_item(mode=mode, item=item):
            return mode
        return f"{mode} {item}"

    @staticmethod
    def _getitem_index(idx):
        return idx

    @staticmethod
    def _getitem_from_ctx(_, ctx, ctx_key):
        return ctx[ctx_key]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(len(self))[idx]]
        if isinstance(idx, list):
            return [self[i] for i in idx]
        if idx < 0:
            idx = len(self) + idx

        items = {}
        ctx = {}
        for key, getitem_fn in self._getitem_fns.items():
            # getitem methods have an optional "ctx" argument -> inspect if ctx is needed
            # isinstance(partial) -> _getitem_from_ctx
            # inspect.getfullargspec(getitem_fn).args -> getitem defines argument "ctx"
            kwargs = {}
            if isinstance(getitem_fn, partial) or "ctx" in inspect.getfullargspec(getitem_fn).args:
                kwargs["ctx"] = ctx
            items[key] = getitem_fn(idx, **kwargs)

        # allow returning as tuple to enable "for x, y in dataloader"
        if self.return_type == dict:
            return items
        elif self.return_type == tuple:
            if len(items) == 1:
                return list(items.values())[0]
            return tuple(items.values())
        raise NotImplementedError(f"invalid return_type '{self.return_type}'")

    def __getattr__(self, item):
        if item == "dataset":
            return getattr(super(), item)
        if item == "__getitems__":
            # new torch versions (>=2.1) implements this which leads to wrappers being circumvented
            # -> disable batched getitems and call getitem instead
            # this occoured when doing DataLoader(dataset) where dataset is ModeWrapper(Subset(...))
            # Subset implements __getitems__ which leads to the fetcher from the DataLoader believing also the
            # ModeWrapper has a __getitems__ and therefore calls it instead of the __getitem__ function
            # returning None makes the DataLoader believe that __getitems__ is not supported
            return None
        return getattr(self.dataset, item)
