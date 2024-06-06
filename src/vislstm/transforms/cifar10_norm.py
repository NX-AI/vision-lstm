from ksuit.data.transforms import ImageMomentNorm


class Cifar10Norm(ImageMomentNorm):
    def __init__(self, **kwargs):
        super().__init__(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616), **kwargs)
