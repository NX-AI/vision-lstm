import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from ksuit.data.transforms import StochasticTransform

from kappadata.utils.magnitude_sampler import MagnitudeSampler


class RandAugment(StochasticTransform):
    """
    reimplementation based on timm
    note that different implementations are vastly different from each other:
    - timm:
      - each transform has a chance of 50% to be applied
        - number of applied transforms is stochastic
        - num_ops is essentially halved
      - magnitude std
      - more transforms than the original (solarize_add, invert)
      - no identity transform (is essentially there because of the 50% apply rate)
    - torchvision:
      - magnitude is in [0, num_magnitude_bins)
        - if the range of a value would be in [0.0, 1.0] magnitude 9 would by default actually result in 0.3 (31 bins)
    - original: https://arxiv.org/abs/1909.13719
      - exactly num_ops operations are sampled
      - ops include identity
    NOTE: it is possible that posterize deletes the whole image
    """

    def __init__(
            self,
            num_ops: int,
            magnitude: int,
            fill_color,
            interpolation: str,
            magnitude_std: float = 0.,
            magnitude_min: float = 0.,
            magnitude_max: float = 10.,
            apply_op_p: float = 0.5,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(num_ops, int) and 0 <= num_ops
        assert isinstance(magnitude, (int, float)) and 0 <= magnitude <= 10
        self.num_ops = num_ops
        if isinstance(interpolation, str):
            if interpolation == "random":
                # timm uses only bilinear or bicubic
                self.interpolations = [InterpolationMode("bilinear"), InterpolationMode("bicubic")]
            else:
                self.interpolations = [InterpolationMode(interpolation)]
        else:
            assert isinstance(interpolation, InterpolationMode)
            self.interpolations = [interpolation]
        self.ops = self._get_ops()
        self.apply_op_p = apply_op_p
        self.magnitude_sampler = MagnitudeSampler(
            magnitude=magnitude / 10,
            magnitude_std=magnitude_std / 10,
            magnitude_min=magnitude_min / 10,
            magnitude_max=magnitude_max / 10,
        )
        self.fill_color = tuple(fill_color)

    def _get_ops(self):
        return [
            # self.identity, # timm applies each transform with 50% probability
            self.auto_contrast,
            self.equalize,
            # not in original publication (but timm uses it)
            self.invert,
            self.rotate,
            self.posterize,
            self.solarize,
            # not in original publication (but timm uses it)
            self.solarize_add,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_horizontal,
            self.translate_vertical,
        ]

    def _sample_transforms(self):
        return self.rng.choice(self.ops, size=self.num_ops)

    def __call__(self, x, ctx=None):
        assert not torch.is_tensor(x), "some KDRandAugment transforms require input to be pillow image"
        transforms = self._sample_transforms()
        for transform in transforms:
            if self.rng.random() < self.apply_op_p:
                x = transform(x, self.magnitude_sampler.sample(self.rng))
        return x

    def _sample_interpolation(self):
        return self.rng.choice(self.interpolations)

    @staticmethod
    def identity(x, _):
        return x

    @staticmethod
    def invert(x, _):
        return F.invert(x)

    @staticmethod
    def auto_contrast(x, _):
        return F.autocontrast(x)

    @staticmethod
    def equalize(x, _):
        return F.equalize(x)

    def rotate(self, x, magnitude):
        # degrees in [-30, 30]
        degrees = 30 * magnitude
        if self.rng.random() > 0.5:
            degrees = -degrees
        return F.rotate(x, degrees, interpolation=self._sample_interpolation(), fill=self.fill_color)

    @staticmethod
    def solarize(x, magnitude):
        # lower threshold -> stronger augmentation
        # threshold >= 256 -> no augmentation
        # threshold in [0, 256]
        threshold = 256 - int(256 * magnitude)
        return F.solarize(x, threshold)

    @staticmethod
    def solarize_add(x, magnitude):
        # higher -> stronger augmentation
        # add in [0, 110]
        # adapted from timm.data.auto_augment.solarize_add
        add = int(magnitude * 110)
        thresh = 128
        lut = []
        for i in range(256):
            if i < thresh:
                lut.append(min(255, i + add))
            else:
                lut.append(i)
        if x.mode == "RGB":
            # repeat for all 3 channels
            lut = lut * 3
        else:
            assert x.mode == "L"
        return x.point(lut)

    def _adjust_factor(self, magnitude):
        offset = 0.9 * magnitude
        if self.rng.random() < 0.5:
            return 1 + offset
        return 1 - offset

    def color(self, x, magnitude):
        # factor == 0 -> black/white image
        # factor == 1 -> identity
        # factor == 2 -> double saturation
        # factor in [0.1, 1.9]
        factor = self._adjust_factor(magnitude)
        return F.adjust_saturation(x, factor)

    @staticmethod
    def posterize(x, magnitude):
        # bits == 0 -> black image
        # bits == 8 -> identity
        # torchvision uses range [4, 8]
        # timm has multiple versions but the RandAug uses [0, 4]
        # timm notes that AutoAugment uses [4, 8] while TF EfficientNet uses [0, 4]
        bits = 4 - int(4 * magnitude)
        return F.posterize(x, bits)

    def contrast(self, x, magnitude):
        # factor == 0 -> solid gray image
        # factor == 1 -> identity
        # factor == 2 -> double contrast
        # factor in [0.1, 1.9]
        factor = self._adjust_factor(magnitude)
        return F.adjust_contrast(x, factor)

    def brightness(self, x, magnitude):
        # factor == 0 -> black image
        # factor == 1 -> identity
        # factor == 2 -> double brightness
        # factor in [0.1, 1.9]
        factor = self._adjust_factor(magnitude)
        return F.adjust_brightness(x, factor)

    def sharpness(self, x, magnitude):
        # factor == 0 -> blurred image
        # factor == 1 -> identity
        # factor == 2 -> double sharpness
        # factor in [0.1, 1.9]
        factor = self._adjust_factor(magnitude)
        return F.adjust_sharpness(x, factor)

    def _shear_degrees(self, magnitude):
        # angle in [-0.3, 0.3]
        angle = 0.3 * magnitude
        # degrees roughly in [-16.7, 16.7]
        if self.rng.random() < 0.5:
            return angle
        return -angle

    def shear_x(self, x, magnitude):
        shear_degrees = self._shear_degrees(magnitude)
        # not sure about the equivalent in torchvision
        # return F.affine(
        #     x,
        #     angle=0.,
        #     translate=[0, 0],
        #     scale=1.,
        #     shear=[shear_degrees, 0.],
        #     interpolation=self._sample_interpolation(),
        #     center=[0, 0],
        #     fill=self.fill_color,
        # )
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, shear_degrees, 0, 0, 1, 0),
            fillcolor=self.fill_color,
            resample=F.pil_modes_mapping[self._sample_interpolation()],
        )

    def shear_y(self, x, magnitude):
        shear_degrees = self._shear_degrees(magnitude)
        # not sure about the equivalent in torchvision
        # return F.affine(
        #     x,
        #     angle=0.,
        #     translate=[0, 0],
        #     scale=1.,
        #     shear=[0., shear_degrees],
        #     interpolation=self._sample_interpolation(),
        #     center=[0, 0],
        #     fill=self.fill_color,
        # )
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, 0, shear_degrees, 1, 0),
            fillcolor=self.fill_color,
            resample=F.pil_modes_mapping[self._sample_interpolation()],
        )

    def _translation(self, magnitude):
        # translation in [-0.45, 0.45]
        translation = 0.45 * magnitude
        if self.rng.random() < 0.5:
            return translation
        return -translation

    def translate_horizontal(self, x, magnitude):
        # PIL image sizes are (width, height)
        # not sure if this should be rounded to int...timm doesn't do it
        translation = self._translation(magnitude) * x.size[0]
        # not sure about the equivalent in torchvision
        # return F.affine(
        #     x,
        #     angle=0.,
        #     translate=[translation, 0],
        #     scale=1.,
        #     interpolation=self._sample_interpolation(),
        #     shear=[0., 0.],
        #     fill=self.fill_color,
        # )
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, translation, 0, 1, 0),
            fillcolor=self.fill_color,
            resample=F.pil_modes_mapping[self._sample_interpolation()],
        )

    def translate_vertical(self, x, magnitude):
        # PIL image sizes are (width, height)
        # not sure if this should be rounded to int...timm doesn't do it
        translation = self._translation(magnitude) * x.size[1]
        # not sure about the equivalent in torchvision
        # return F.affine(
        #     x,
        #     angle=0.,
        #     translate=[0, translation],
        #     scale=1.,
        #     interpolation=self._sample_interpolation(),
        #     shear=[0., 0.],
        #     fill=self.fill_color,
        # )
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, translation),
            fillcolor=self.fill_color,
            resample=F.pil_modes_mapping[self._sample_interpolation()],
        )
