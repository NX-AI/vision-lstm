import numpy as np
from torch import nn

from .base import SingleModel


class LinearModel(SingleModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        input_dim = np.prod(self.input_shape)
        output_dim = np.prod(self.output_shape)
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x.flatten(start_dim=1)).reshape(len(x), *self.output_shape)

    def classify(self, x):
        return dict(main=self.forward(x))
