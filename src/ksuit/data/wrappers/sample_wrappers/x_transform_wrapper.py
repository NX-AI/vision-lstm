from ksuit.data.utils.optional_kwargs import optional_ctx
from .base.transform_wrapper_base import TransformWrapperBase


class XTransformWrapper(TransformWrapperBase):
    def getitem_x(self, idx, ctx=None):
        item = self.dataset.getitem_x(idx, **optional_ctx(fn=self.dataset.getitem_x, ctx=ctx))
        return self._getitem(item=item, idx=idx, ctx=ctx)
