import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.upernet.modeling_upernet import UperNetPyramidPoolingModule, UperNetConvModule

from ksuit.models import SingleModel


class UperNetDecoder(SingleModel):
    # adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/upernet/modeling_upernet.py
    def __init__(self, input_dim, dim=512, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        num_inputs = 4
        self.pool_scales = [1, 2, 3, 6]
        self.dim = dim
        self.align_corners = False
        self.dropout = dropout

        # PSP Module
        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            input_dim,
            dim,
            align_corners=self.align_corners,
        )
        self.bottleneck = UperNetConvModule(
            input_dim + len(self.pool_scales) * dim,
            dim,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for _ in range(num_inputs - 1):  # skip the top layer
            l_conv = UperNetConvModule(input_dim, dim, kernel_size=1)
            fpn_conv = UperNetConvModule(dim, dim, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = UperNetConvModule(
            num_inputs * dim,
            dim,
            kernel_size=3,
            padding=1,
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

    def forward(self, encoder_hidden_states):
        # build laterals
        laterals = [
            lateral_conv(encoder_hidden_states[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # psp_forward
        psp_result = self.bottleneck(
            torch.concat(
                [
                    encoder_hidden_states[-1],
                    *self.psp_modules(encoder_hidden_states[-1])
                ],
                dim=1,
            ),
        )
        laterals.append(psp_result)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = torch.concat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(self.dropout(output))

        output = F.interpolate(output, size=self.output_shape[1:], mode="bilinear", align_corners=False)

        return output
