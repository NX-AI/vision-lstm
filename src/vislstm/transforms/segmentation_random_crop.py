from torchvision.transforms.functional import get_image_size, crop

from ksuit.data.transforms import StochasticTransform
from ksuit.utils.param_checking import to_2tuple


class SegmentationRandomCrop(StochasticTransform):
    def __init__(self, size, max_category_ratio=1., ignore_index=-1, **kwargs):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)
        self.max_category_ratio = max_category_ratio
        self.ignore_index = ignore_index

    def __call__(self, xseg, ctx=None):
        x, seg = xseg
        width, height = get_image_size(x)

        top, left, crop_height, crop_width = self.get_params(height=height, width=width)
        semseg_crop = crop(seg, top, left, crop_height, crop_width)
        if self.max_category_ratio < 1.:
            for _ in range(10):
                labels, counts = semseg_crop.unique(return_counts=True)
                counts = counts[labels != self.ignore_index]
                if len(counts) > 1 and counts.max() / counts.sum() < self.max_category_ratio:
                    break
                top, left, crop_height, crop_width = self.get_params(height=height, width=width)
                semseg_crop = crop(seg, top, left, crop_height, crop_width)

        x = crop(x, top, left, crop_height, crop_width)
        seg = semseg_crop
        return x, seg

    def get_params(self, height, width):
        top = int(self.rng.integers(max(0, height - self.size[0]) + 1, size=(1,)))
        left = int(self.rng.integers(max(0, width - self.size[1]) + 1, size=(1,)))
        crop_height = min(height, self.size[0])
        crop_width = min(width, self.size[1])
        return top, left, crop_height, crop_width
