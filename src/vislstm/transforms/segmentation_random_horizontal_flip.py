from torchvision.transforms.functional import hflip

from ksuit.data.transforms import StochasticTransform


class SegmentationRandomHorizontalFlip(StochasticTransform):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def __call__(self, xseg, ctx=None):
        apply = self.rng.random() < self.p
        x, seg = xseg
        if apply:
            x = hflip(x)
            seg = hflip(seg)
        return x, seg
