import torch

from ksuit.data import Wrapper
import inspect

class LabelSmoothingWrapper(Wrapper):
    def __init__(self, dataset, smoothing):
        super().__init__(dataset=dataset)
        assert isinstance(smoothing, (int, float)) and 0. <= smoothing <= 1.
        self.smoothing = smoothing

    def getitem_class(self, idx, ctx=None):
        kwargs = {}
        if "ctx" in inspect.getfullargspec(self.dataset.getitem_class).args:
            kwargs["ctx"] = ctx
        y = self.dataset.getitem_class(idx, **kwargs)
        if self.smoothing == 0:
            return y
        assert isinstance(y, int) or (torch.is_tensor(y) and y.ndim == 0)
        n_classes = self.dataset.getdim_class()

        # semi supervised case (can't smooth missing labels)
        if y == -1:
            return torch.full(size=(n_classes,), fill_value=-1.)

        # binary case (label is scalar)
        if n_classes == 1:
            off_value = self.smoothing / 2
            if y > 0.5:
                return y - off_value
            else:
                return y + off_value

        # multi class (scalar -> vector)
        off_value = self.smoothing / n_classes
        on_value = 1. - self.smoothing + off_value
        y_vector = torch.full(size=(n_classes,), fill_value=off_value)
        y_vector[y] = on_value
        return y_vector
