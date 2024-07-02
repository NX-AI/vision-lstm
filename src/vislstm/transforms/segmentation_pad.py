from torchvision.transforms.functional import pad, get_image_size

from ksuit.data.transforms import Transform
from ksuit.utils.param_checking import to_2tuple


class SegmentationPad(Transform):
    def __init__(self, size, fill=-1, **kwargs):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)
        self.fill = fill

    def __call__(self, xseg, ctx=None):
        x, seg = xseg
        width, height = get_image_size(x)
        pad_h = max(0, self.size[0] - height)
        pad_w = max(0, self.size[1] - width)
        pad_top = pad_bot = pad_h // 2
        if pad_h % 2 == 1:
            pad_bot += 1
        pad_left = pad_right = pad_w // 2
        if pad_w % 2 == 1:
            pad_right += 1

        padding = (pad_left, pad_top, pad_right, pad_bot)
        x = pad(x, padding=padding, fill=0)
        seg = pad(seg, padding=padding, fill=self.fill)
        return x, seg
