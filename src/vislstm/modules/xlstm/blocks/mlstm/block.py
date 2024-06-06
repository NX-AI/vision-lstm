# This file is licensed under AGPL-3.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass, field

from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import mLSTMLayerConfig


@dataclass
class mLSTMBlockConfig:
    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)

    def __post_init__(self):
        self.mlstm.__post_init__()
        return self


class mLSTMBlock(xLSTMBlock):

    config_class = mLSTMBlockConfig

    def __init__(self, config: mLSTMBlockConfig) -> None:
        super().__init__(config=xLSTMBlockConfig(mlstm=config.mlstm, slstm=None, feedforward=None))
