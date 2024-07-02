import torch.nn as nn

from ksuit.models import SingleModel
from kappamodules.init import init_xavier_uniform_zero_bias


class UpernetPostprocessor(SingleModel):
    """ https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/backbone/beit.py#L418 """

    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        patch_size = self.static_ctx["patch_size"]
        dim = input_dim
        if patch_size == (16, 16) or patch_size == (14, 14):
            fpn1 = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            )
            fpn2 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
            fpn3 = nn.Identity()
            fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == (8, 8):
            fpn1 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
            fpn2 = nn.Identity()
            fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)
            fpn4 = nn.MaxPool2d(kernel_size=4, stride=4)
        else:
            raise NotImplementedError
        self.processors = nn.ModuleList([fpn1, fpn2, fpn3, fpn4])
        init_xavier_uniform_zero_bias(self)

    def forward(self, x):
        assert len(x) == len(self.processors)
        return [processor(xx) for xx, processor in zip(x, self.processors)]
