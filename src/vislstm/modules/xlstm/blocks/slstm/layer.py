# This file is licensed under AGPL-3.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel
from dataclasses import dataclass
from typing import Optional
import torch
from ...components.ln import MultiHeadLayerNorm
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import small_init_init_

from torch import nn
from .cell import sLSTMCell, sLSTMCellConfig


@dataclass
class sLSTMLayerConfig(sLSTMCellConfig):
    embedding_dim: int = -1
    num_heads: int = 4  # this must divide the hidden size, is not yet supported by all versions in this directory
    conv1d_kernel_size: int = 4  # 0 means no convolution included

    dropout: float = 0.0

    def __post_init__(self):
        self.hidden_size = self.embedding_dim
        sLSTMCellConfig.__post_init__(self)
        return self


class sLSTMLayer(nn.Module):
    config_class = sLSTMLayerConfig

    def __init__(self, config: sLSTMLayerConfig):
        super().__init__()
        self.config = config

        if self.config.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(
                config=CausalConv1dConfig(
                    feature_dim=self.config.embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                )
            )
            self.conv_act_fn = nn.SiLU()

        self.fgate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.igate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.zgate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.ogate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )

        self.slstm_cell = sLSTMCell(self.config)
        self.group_norm = MultiHeadLayerNorm(ndim=self.config.embedding_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def reset_parameters(self):
        self.slstm_cell.reset_parameters()
        self.group_norm.reset_parameters()
        small_init_init_(self.igate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.fgate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.zgate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.ogate.weight, dim=self.config.embedding_dim)

    def forward(
        self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None, return_last_state=False, **kwargs
    ) -> torch.Tensor:
        B, S, _ = x.shape

        if self.config.conv1d_kernel_size > 0:
            x_conv = self.conv_act_fn(self.conv1d(x))
        else:
            x_conv = x

        i, f, z, o = self.fgate(x_conv), self.igate(x_conv), self.zgate(x), self.ogate(x)

        y, last_state = self.slstm_cell(torch.cat([i, f, z, o], dim=-1), state=initial_state)

        y = self.dropout(y)

        out = self.group_norm(y).transpose(1, 2).view(B, S, -1)

        if return_last_state:
            return out, last_state
        else:
            return out
