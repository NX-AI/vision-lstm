# This file is licensed under AGPL-3.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Benedikt Alkin
from dataclasses import dataclass

import einops
import torch
from kappamodules.layers import LayerScale
from torch import nn

from .cell import mLSTMCell, mLSTMCellConfig
from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import small_init_init_, wang_init_
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...utils import UpProjConfigMixin
import math

@dataclass
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 2.0
    bidirectional: bool = False
    quaddirectional: bool = False
    sharedirs: bool = False
    alternation: str = None
    layerscale: float = None

    # will be set toplevel config
    embedding_dim: int = -1
    bias: bool = False
    dropout: float = 0.0
    context_length: int = -1

    _num_blocks: int = 1
    _inner_embedding_dim: int = None

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        self._inner_embedding_dim = self._proj_up_dim
        return self


class mLSTMLayer(nn.Module):
    config_class = mLSTMLayerConfig

    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config

        print(self.config)
        self.proj_up = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=2 * self.config._inner_embedding_dim,
            bias=self.config.bias,
        )

        num_proj_heads = round(self.config._inner_embedding_dim // self.config.qkv_proj_blocksize)
        self.q_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )
        self.k_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )
        self.v_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )

        self.conv1d = CausalConv1d(
            config=CausalConv1dConfig(
                feature_dim=self.config._inner_embedding_dim,
                kernel_size=self.config.conv1d_kernel_size,
            )
        )
        self.conv_act_fn = nn.SiLU()
        self.mlstm_cell = mLSTMCell(
            config=mLSTMCellConfig(
                context_length=self.config.context_length,
                embedding_dim=self.config._inner_embedding_dim,
                num_heads=self.config.num_heads,
            )
        )
        self.ogate_act_fn = nn.SiLU()

        self.learnable_skip = nn.Parameter(torch.ones(self.config._inner_embedding_dim, requires_grad=True))

        # bidirectional
        if (self.config.bidirectional or self.config.quaddirectional) and not self.config.sharedirs:
            self.q_proj_rev = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.k_proj_rev = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.v_proj_rev = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.conv1d_rev = CausalConv1d(
                config=CausalConv1dConfig(
                    feature_dim=self.config._inner_embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                )
            )
            self.mlstm_cell_rev = mLSTMCell(
                config=mLSTMCellConfig(
                    context_length=self.config.context_length,
                    embedding_dim=self.config._inner_embedding_dim,
                    num_heads=self.config.num_heads,
                )
            )
            self.learnable_skip_rev = nn.Parameter(torch.ones(self.config._inner_embedding_dim, requires_grad=True))
        else:
            self.q_proj_rev = None
            self.k_proj_rev = None
            self.v_proj_rev = None
            self.conv1d_rev = None
            self.mlstm_cell_rev = None
            self.learnable_skip_rev = None

        if self.config.quaddirectional and not self.config.sharedirs:
            self.q_proj_ud = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.k_proj_ud = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.v_proj_ud = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.conv1d_ud = CausalConv1d(
                config=CausalConv1dConfig(
                    feature_dim=self.config._inner_embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                )
            )
            self.mlstm_cell_ud = mLSTMCell(
                config=mLSTMCellConfig(
                    context_length=self.config.context_length,
                    embedding_dim=self.config._inner_embedding_dim,
                    num_heads=self.config.num_heads,
                )
            )
            self.learnable_skip_ud = nn.Parameter(torch.ones(self.config._inner_embedding_dim, requires_grad=True))
            self.q_proj_du = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.k_proj_du = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.v_proj_du = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                )
            )
            self.conv1d_du = CausalConv1d(
                config=CausalConv1dConfig(
                    feature_dim=self.config._inner_embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                )
            )
            self.mlstm_cell_du = mLSTMCell(
                config=mLSTMCellConfig(
                    context_length=self.config.context_length,
                    embedding_dim=self.config._inner_embedding_dim,
                    num_heads=self.config.num_heads,
                )
            )
            self.learnable_skip_du = nn.Parameter(torch.ones(self.config._inner_embedding_dim, requires_grad=True))
        else:
            self.q_proj_ud = None
            self.k_proj_ud = None
            self.v_proj_ud = None
            self.conv1d_ud = None
            self.mlstm_cell_ud = None
            self.learnable_skip_ud = None
            self.q_proj_du = None
            self.k_proj_du = None
            self.v_proj_du = None
            self.conv1d_du = None
            self.mlstm_cell_du = None
            self.learnable_skip_du = None

        self.proj_down = nn.Linear(
            in_features=self.config._inner_embedding_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.bias,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        if self.config.layerscale is None:
            self.layerscale = nn.Identity()
        else:
            self.layerscale = LayerScale(dim=self.config.embedding_dim, init_scale=self.config.layerscale)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, block_idx=None, **kwargs) -> torch.Tensor:
        B, S, _ = x.shape

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.split(x_inner, split_size_or_sections=self.config._inner_embedding_dim, dim=-1)

        # alternate direction in successive layers
        if self.config.alternation is None:
            pass
        elif self.config.alternation == "bidirectional":
            if block_idx % 2 == 1:
                x_mlstm = x_mlstm.flip(dims=[1])
                z = z.flip(dims=[1])
            else:
                pass
        elif self.config.alternation == "quaddirectional":
            assert x_mlstm.size(1) % 2 == 0
            w = int(math.sqrt(x_mlstm.size(1)))
            if block_idx % 4 == 1:
                x_mlstm = x_mlstm.flip(dims=[1])
                z = z.flip(dims=[1])
            if block_idx % 4 == 2:
                x_mlstm = einops.rearrange(x_mlstm, "b (h w) d -> b (w h) d", w=w)
                z = einops.rearrange(z, "b (h w) d -> b (w h) d", w=w)
            if block_idx % 4 == 3:
                x_mlstm = einops.rearrange(x_mlstm, "b (h w) d -> b (w h) d", w=w).flip(dims=[1])
                z = einops.rearrange(z, "b (h w) d -> b (w h) d", w=w).flip(dims=[1])
            else:
                pass
        else:
            raise NotImplementedError

        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = self.conv_act_fn(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # reverse alternating flip
        if self.config.alternation is None:
            pass
        elif self.config.alternation == "bidirectional":
            if block_idx % 2 == 1:
                h_state = h_state.flip(dims=[1])
            else:
                pass
        elif self.config.alternation == "quaddirectional":
            assert h_state.size(1) % 2 == 0
            w = int(math.sqrt(x_mlstm.size(1)))
            if block_idx % 4 == 1:
                h_state = h_state.flip(dims=[1])
            if block_idx % 4 == 2:
                h_state = einops.rearrange(h_state, "b (w h) d -> b (h w) d", w=w)
            if block_idx % 4 == 3:
                h_state = einops.rearrange(h_state, "b (w h) d -> b (h w) d", w=w).flip(dims=[1])
            else:
                pass
        else:
            raise NotImplementedError

        if self.config.bidirectional or self.config.quaddirectional:
            if self.config.sharedirs:
                conv1d_rev = self.conv1d
                q_proj_rev = self.q_proj
                k_proj_rev = self.k_proj
                v_proj_rev = self.v_proj
                mlstm_cell_rev = self.mlstm_cell
                learnable_skip_rev = self.learnable_skip
            else:
                conv1d_rev = self.conv1d_rev
                q_proj_rev = self.q_proj_rev
                k_proj_rev = self.k_proj_rev
                v_proj_rev = self.v_proj_rev
                mlstm_cell_rev = self.mlstm_cell_rev
                learnable_skip_rev = self.learnable_skip_rev
            x_mlstm_rev = x_mlstm.flip(dims=[1])
            z_rev = z.flip(dims=[1])
            x_mlstm_conv_rev = conv1d_rev(x_mlstm_rev)
            x_mlstm_conv_act_rev = self.conv_act_fn(x_mlstm_conv_rev)

            q_rev = q_proj_rev(x_mlstm_conv_act_rev)
            k_rev = k_proj_rev(x_mlstm_conv_act_rev)
            v_rev = v_proj_rev(x_mlstm_rev)

            h_tilde_state_rev = mlstm_cell_rev(q=q_rev, k=k_rev, v=v_rev)
            h_tilde_state_skip_rev = h_tilde_state_rev + (learnable_skip_rev * x_mlstm_conv_act_rev)

            h_state = h_state + (h_tilde_state_skip_rev * self.ogate_act_fn(z_rev)).flip(dims=[1])

        if self.config.quaddirectional:
            if self.config.sharedirs:
                conv1d_du = self.conv1d
                q_proj_du = self.q_proj
                k_proj_du = self.k_proj
                v_proj_du = self.v_proj
                mlstm_cell_du = self.mlstm_cell
                learnable_skip_du = self.learnable_skip
                conv1d_ud = self.conv1d
                q_proj_ud = self.q_proj
                k_proj_ud = self.k_proj
                v_proj_ud = self.v_proj
                mlstm_cell_ud = self.mlstm_cell
                learnable_skip_ud = self.learnable_skip
            else:
                conv1d_du = self.conv1d_du
                q_proj_du = self.q_proj_du
                k_proj_du = self.k_proj_du
                v_proj_du = self.v_proj_du
                mlstm_cell_du = self.mlstm_cell_du
                learnable_skip_du = self.learnable_skip_du
                conv1d_ud = self.conv1d_ud
                q_proj_ud = self.q_proj_ud
                k_proj_ud = self.k_proj_ud
                v_proj_ud = self.v_proj_ud
                mlstm_cell_ud = self.mlstm_cell_ud
                learnable_skip_ud = self.learnable_skip_ud

            assert x_mlstm.size(1) % 2 == 0
            w = int(math.sqrt(x_mlstm.size(1)))
            x_mlstm_du = einops.rearrange(x_mlstm, "b (h w) d -> b (w h) d", w=w)
            z_du = einops.rearrange(z, "b (h w) d -> b (w h) d", w=w)
            x_mlstm_conv_du = conv1d_du(x_mlstm_du)
            x_mlstm_conv_act_du = self.conv_act_fn(x_mlstm_conv_du)
            q_du = q_proj_du(x_mlstm_conv_act_du)
            k_du = k_proj_du(x_mlstm_conv_act_du)
            v_du = v_proj_du(x_mlstm_du)
            h_tilde_state_du = mlstm_cell_du(q=q_du, k=k_du, v=v_du)
            h_tilde_state_skip_du = h_tilde_state_du + (learnable_skip_du * x_mlstm_conv_act_du)
            out_du = einops.rearrange(h_tilde_state_skip_du * self.ogate_act_fn(z_du), "b (w h) d -> b (h w) d", w=w)
            h_state = h_state + out_du

            x_mlstm_ud = einops.rearrange(x_mlstm, "b (h w) d -> b (w h) d", w=w).flip(dims=[1])
            z_ud = einops.rearrange(z, "b (h w) d -> b (w h) d", w=w).flip(dims=[1])
            x_mlstm_conv_ud = conv1d_ud(x_mlstm_ud)
            x_mlstm_conv_act_ud = self.conv_act_fn(x_mlstm_conv_ud)
            q_ud = q_proj_ud(x_mlstm_conv_act_ud)
            k_ud = k_proj_ud(x_mlstm_conv_act_ud)
            v_ud = v_proj_ud(x_mlstm_ud)
            h_tilde_state_ud = mlstm_cell_ud(q=q_ud, k=k_ud, v=v_ud)
            h_tilde_state_skip_ud = h_tilde_state_ud + (learnable_skip_ud * x_mlstm_conv_act_ud)
            out_ud = einops.rearrange(h_tilde_state_skip_ud * self.ogate_act_fn(z_ud), "b (w h) d -> b (h w) d", w=w)
            h_state = h_state + out_ud.flip(dims=[1])

        # down-projection
        y = self.dropout(self.proj_down(h_state))

        # layerscale
        y = self.layerscale(y)

        return y

    def reset_parameters(self):
        # init inproj
        small_init_init_(self.proj_up.weight, dim=self.config.embedding_dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj
        wang_init_(self.proj_down.weight, dim=self.config.embedding_dim, num_blocks=self.config._num_blocks)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)
        if self.learnable_skip_rev is not None:
            nn.init.ones_(self.learnable_skip_rev)
        if self.learnable_skip_du is not None:
            nn.init.ones_(self.learnable_skip_du)
        if self.learnable_skip_ud is not None:
            nn.init.ones_(self.learnable_skip_ud)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            if qkv_proj is None:
                return
            # use the embedding dim instead of the inner embedding dim
            small_init_init_(qkv_proj.weight, dim=self.config.embedding_dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)
        _init_qkv_proj(self.q_proj_rev)
        _init_qkv_proj(self.k_proj_rev)
        _init_qkv_proj(self.v_proj_rev)
        _init_qkv_proj(self.q_proj_du)
        _init_qkv_proj(self.k_proj_du)
        _init_qkv_proj(self.v_proj_du)
        _init_qkv_proj(self.q_proj_ud)
        _init_qkv_proj(self.k_proj_ud)
        _init_qkv_proj(self.v_proj_ud)

        self.mlstm_cell.reset_parameters()
        if self.mlstm_cell_rev is not None:
            self.mlstm_cell_rev.reset_parameters()
        if self.mlstm_cell_du is not None:
            self.mlstm_cell_du.reset_parameters()
        if self.mlstm_cell_ud is not None:
            self.mlstm_cell_ud.reset_parameters()
