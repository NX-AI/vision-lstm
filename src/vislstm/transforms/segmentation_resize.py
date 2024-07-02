import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode, resize

from ksuit.data.transforms import Transform


class SegmentationResize(Transform):
    def __init__(self, *args, interpolation="bilinear", **kwargs):
        super().__init__()
        self.resize = Resize(*args, interpolation=InterpolationMode(interpolation), antialias=True, **kwargs)

    def __call__(self, xseg, ctx=None):
        # checks
        assert isinstance(xseg, tuple) and len(xseg) == 2

        # setup
        x, seg = xseg
        squeeze = False
        if torch.is_tensor(seg):
            seg = seg.unsqueeze(0)
            squeeze = True

        # transform
        x = self.resize(x)
        seg = resize(img=seg, size=self.resize.size, interpolation=InterpolationMode.NEAREST)

        # cleanup
        if squeeze:
            seg = seg.squeeze(0)
        return x, seg
