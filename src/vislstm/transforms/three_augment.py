import torchvision.transforms.functional as F
from PIL import ImageOps

from ksuit.data.transforms import StochasticTransform
from .gaussian_blur_pil import GaussianBlurPIL


class ThreeAugment(StochasticTransform):
    def __init__(self, blur_sigma=(0.1, 2.0), **kwargs):
        super().__init__(**kwargs)
        self.gaussian_blur = GaussianBlurPIL(sigma=blur_sigma)

    def set_rng(self, rng):
        self.gaussian_blur.set_rng(rng)
        return super().set_rng(rng)

    def __call__(self, x, ctx=None):
        choice = self.rng.integers(3)
        if choice == 0:
            return F.rgb_to_grayscale(x, num_output_channels=F.get_image_num_channels(x))
        if choice == 1:
            return ImageOps.solarize(x)
        if choice == 2:
            return self.gaussian_blur(x)
        raise NotImplementedError
