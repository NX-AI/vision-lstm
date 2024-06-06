# This file is licensed under AGPL-3.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Benedikt Alkin
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from kappamodules.layers import DropPath
from torch import nn

from .mlstm.layer import mLSTMLayer, mLSTMLayerConfig
from .slstm.layer import sLSTMLayerConfig
from ..components.feedforward import FeedForwardConfig, create_feedforward
from ..components.ln import LayerNorm

"""An xLSTM block can be either an sLSTM Block or an mLSTM Block.

It contains the pre-LayerNorms and the skip connections.
"""


@dataclass
class xLSTMBlockConfig:
    mlstm: Optional[mLSTMLayerConfig] = None
    slstm: Optional[sLSTMLayerConfig] = None

    feedforward: Optional[FeedForwardConfig] = None
    drop_path: float = 0.
    _num_blocks: int = 1
    _block_idx: int = 0

    def __post_init__(self):
        assert (
            self.mlstm is not None or self.slstm is not None
        ), "Either mlstm or slstm must be provided"
        assert (
            self.mlstm is None or self.slstm is None
        ), "Only one of mlstm or slstm can be provided"
        embedding_dim = (
            self.mlstm.embedding_dim
            if self.mlstm is not None
            else self.slstm.embedding_dim
        )
        if self.mlstm:
            self.mlstm._num_blocks = self._num_blocks
            self.mlstm._block_idx = self._block_idx
        if self.slstm:
            self.slstm._num_blocks = self._num_blocks
            self.slstm._block_idx = self._block_idx
        if self.feedforward:
            self.feedforward.embedding_dim = embedding_dim
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.__post_init__()
        return self


class xLSTMBlock(nn.Module):

    config_class = xLSTMBlockConfig

    def __init__(self, config: xLSTMBlockConfig) -> None:
        super().__init__()
        self.config = config
        embedding_dim = (
            self.config.mlstm.embedding_dim
            if self.config.mlstm is not None
            else self.config.slstm.embedding_dim
        )

        self.drop_path1 = DropPath(drop_prob=config.drop_path)
        self.xlstm_norm = LayerNorm(ndim=embedding_dim, weight=True, bias=False)

        if self.config.mlstm is not None:
            self.xlstm = mLSTMLayer(config=self.config.mlstm)
        # elif self.config.slstm is not None:
        #     self.xlstm = sLSTMLayer(config=self.config.slstm)
        else:
            raise ValueError("Either mlstm or slstm must be provided")

        if self.config.feedforward is not None:
            self.ffn_norm = LayerNorm(
                ndim=self.config.feedforward.embedding_dim, weight=True, bias=False
            )
            self.ffn = create_feedforward(config=self.config.feedforward)
        else:
            self.ffn_norm = None
            self.ffn = None

        self.reset_parameters()

    def _xlstm_forward_path(self, x, **kwargs):
        return self.xlstm(self.xlstm_norm(x), **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.drop_path1(x, partial(self._xlstm_forward_path, **kwargs))
        if self.ffn is not None:
            raise NotImplementedError("stochastic depth doesnt support **kwargs yet")
            # x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x

    def reset_parameters(self) -> None:
        self.xlstm.reset_parameters()
        self.xlstm_norm.reset_parameters()
        if self.ffn is not None:
            self.ffn.reset_parameters()
            self.ffn_norm.reset_parameters()
