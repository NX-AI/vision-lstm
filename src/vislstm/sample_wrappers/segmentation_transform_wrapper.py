from ksuit.data.utils.optional_kwargs import optional_ctx
from ksuit.data.wrappers import TransformWrapperBase
import uuid

class SegmentationTransformWrapper(TransformWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx_prefix = uuid.uuid4()

    def _getitem_cached(self, idx, ctx, item):
        assert ctx is not None
        ctx_item = f"{self.ctx_prefix}.{item}"
        if ctx_item in ctx:
            return ctx[ctx_item]

        # load x and segmentation
        x = self.dataset.getitem_x(idx, **optional_ctx(fn=self.dataset.getitem_x, ctx=ctx))
        seg = self.dataset.getitem_segmentation(idx, **optional_ctx(fn=self.dataset.getitem_segmentation, ctx=ctx))

        # apply transform
        x, seg = self._getitem((x, seg), idx=idx, ctx=ctx)

        # cache
        ctx[f"{self.ctx_prefix}.x"] = x
        ctx[f"{self.ctx_prefix}.segmentation"] = seg
        if item == "x":
            return x
        if item == "segmentation":
            return seg
        raise NotImplementedError

    def getitem_x(self, idx, ctx=None):
        return self._getitem_cached(idx=idx, ctx=ctx, item="x")

    def getitem_segmentation(self, idx, ctx=None):
        return self._getitem_cached(idx=idx, ctx=ctx, item="segmentation")
