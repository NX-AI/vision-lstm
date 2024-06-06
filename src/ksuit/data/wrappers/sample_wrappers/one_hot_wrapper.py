from ksuit.data import Wrapper
from ksuit.utils.one_hot_utils import to_one_hot_vector


class OneHotWrapper(Wrapper):
    def getitem_class(self, idx, ctx=None):
        y = self.dataset.getitem_class(idx, ctx)
        return to_one_hot_vector(y, n_classes=self.dataset.getdim_class())
