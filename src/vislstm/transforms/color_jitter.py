import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter as TVColorJitter

from ksuit.data.transforms import StochasticTransform


class ColorJitter(StochasticTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, **kwargs):
        super().__init__(**kwargs)
        # ColorJitter preprocesses the parameters
        tv_colorjitter = TVColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        # store for scaling strength
        if tv_colorjitter.brightness is not None:
            self.brightness_lb = self.og_brightness_lb = tv_colorjitter.brightness[0]
            self.brightness_ub = self.og_brightness_ub = tv_colorjitter.brightness[1]
        else:
            self.brightness_lb = None
        if tv_colorjitter.contrast is not None:
            self.contrast_lb = self.og_contrast_lb = tv_colorjitter.contrast[0]
            self.contrast_ub = self.og_contrast_ub = tv_colorjitter.contrast[1]
        else:
            self.contrast_lb = None
        if tv_colorjitter.saturation is not None:
            self.saturation_lb = self.og_saturation_lb = tv_colorjitter.saturation[0]
            self.saturation_ub = self.og_saturation_ub = tv_colorjitter.saturation[1]
        else:
            self.saturation_lb = None
        if tv_colorjitter.hue is not None:
            self.hue_lb = self.og_hue_lb = tv_colorjitter.hue[0]
            self.hue_ub = self.og_hue_ub = tv_colorjitter.hue[1]
        else:
            self.hue_lb = None

    def __call__(self, x, ctx=None):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params()
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                x = F.adjust_brightness(x, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                x = F.adjust_contrast(x, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                x = F.adjust_saturation(x, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                x = F.adjust_hue(x, hue_factor)
        return x

    def get_params(self):
        fn_idx = self.rng.permutation(4)

        b = None if self.brightness_lb is None else self.rng.uniform(self.brightness_lb, self.brightness_ub)
        c = None if self.contrast_lb is None else self.rng.uniform(self.contrast_lb, self.contrast_ub)
        s = None if self.saturation_lb is None else self.rng.uniform(self.saturation_lb, self.saturation_ub)
        h = None if self.hue_lb is None else self.rng.uniform(self.hue_lb, self.hue_ub)

        return fn_idx, b, c, s, h
