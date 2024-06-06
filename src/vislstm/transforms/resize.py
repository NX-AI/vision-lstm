from torchvision.transforms import Resize as TVResize
from torchvision.transforms.functional import InterpolationMode

from ksuit.data.transforms import Transform


class Resize(Transform):
    """ wrapper for torchvision.transforms.Resize as it doesn't support passing a string as interpolation """

    def __init__(self, *args, interpolation="bilinear", **kwargs):
        super().__init__()
        self.resize = TVResize(*args, interpolation=InterpolationMode(interpolation), antialias=True, **kwargs)

    def __call__(self, x):
        return self.resize(x)
