import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.upernet.modeling_upernet import UperNetConvModule

from ksuit.models import SingleModel


class UpernetAuxiliary(SingleModel):
    # adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/upernet/modeling_upernet.py
    def __init__(
            self,
            input_dim: int,
            dim: int = 256,
            kernel_size: int = 3,
            dilation: int = 1,
            concat_input: bool = False,
            num_convs: int = 1,
            dropout: float = 0.,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.dropout = dropout

        conv_padding = (kernel_size // 2) * dilation
        convs = [
            UperNetConvModule(
                in_channels=input_dim,
                out_channels=dim,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
            )
        ]
        for i in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                )
            )
        self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                in_channels=input_dim + dim,
                out_channels=dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        output_dim, _, _ = self.output_shape
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(dim, output_dim, kernel_size=1)

        def _init_weights(module):
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(_init_weights)

    def forward(self, hidden_states):
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.concat([hidden_states, output], dim=1))
        output = self.classifier(self.dropout(output))
        output = F.interpolate(output, size=self.output_shape[1:], mode="bilinear", align_corners=False)
        return output
