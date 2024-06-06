import math

import numpy as np
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import resized_crop, get_image_size

from ksuit.data.transforms import StochasticTransform
from ksuit.utils.param_checking import to_2tuple


class RandomResizedCrop(StochasticTransform):
    """ wrapper for torchvision.transforms.Resize as it doesn't support passing a string as interpolation """

    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation="bilinear",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = to_2tuple(size)
        self.scale = scale
        self.ratio = ratio
        # RandomResizedCrop doesn't support interpolation argument as string
        self.interpolation = InterpolationMode(interpolation)

    def __call__(self, x):
        i, j, h, w = self.get_params(x)
        return resized_crop(x, i, j, h, w, self.size, self.interpolation)

    def get_params(self, img):
        # same as torchvision.transform.RandomResizedCrop but with rng
        width, height = get_image_size(img)
        area = height * width

        # the torch version has a floating point error which is bad for testing
        # log_ratio = torch.log(torch.tensor(self.ratio))
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = area * self.rng.uniform(self.scale[0], self.scale[1])
            aspect_ratio = np.exp(self.rng.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = int(self.rng.integers(0, height - h + 1))
                j = int(self.rng.integers(0, width - w + 1))
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
