from ksuit.data.transforms import ImageMomentNorm


class Imagenet1kNorm(ImageMomentNorm):
    def __init__(self, **kwargs):
        super().__init__(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs)
