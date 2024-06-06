import numpy as np
import einops
import torch
from kappamodules.functional.pos_embed import interpolate_sincos
from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d, VitClassTokens
from torch import nn

from vislstm.modules.xlstm import create_block_stack
from ksuit.factory import MasterFactory
from ksuit.models import SingleModel
from ksuit.optim.param_group_modifiers import WeightDecayByNameModifier
from ksuit.utils.formatting_utils import list_to_string
from ksuit.utils.param_checking import to_ntuple

class VisLSTM(SingleModel):
    def __init__(
            self,
            patch_size,
            dim,
            depth,
            stride=None,
            cls_tokens=None,
            pos_embed_mode="fixed",
            init_weights="xavier_uniform",
            bidirectional=False,
            quaddirectional=False,
            sharedirs=False,
            mode=None,
            layerscale=None,
            pooling=None,
            alternation=None,
            dropout_rate=0.0,
            drop_path_rate=0.0,
            drop_path_decay=True,
            proj_factor=2.0,
            add_post_blocks_norm=True,
            disable_mask=False,
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        ndim = len(self.input_shape) - 1
        self.patch_size = to_ntuple(patch_size, n=ndim)
        self.static_ctx["patch_size"] = self.patch_size
        self.dim = dim
        self.depth = depth
        self.mode = mode
        self.pooling = MasterFactory.get("pooling").create(pooling, static_ctx=self.static_ctx)
        self.eps = eps
        self.pos_embed_mode = pos_embed_mode
        self.bidirectional = bidirectional
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.proj_factor = proj_factor

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            stride=stride,
            num_channels=self.input_shape[0],
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
            init_weights=init_weights,
        )
        self.static_ctx["sequence_lengths"] = self.patch_embed.seqlens

        # pos embed
        if pos_embed_mode == "learnable":
            self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim, is_learnable=True)
            self.logger.info(f"pos_embed.is_learnable={self.pos_embed.is_learnable}")
        elif pos_embed_mode == "fixed":
            self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim, is_learnable=False)
            self.logger.info(f"pos_embed.is_learnable={self.pos_embed.is_learnable}")
        elif pos_embed_mode == "none":
            self.pos_embed = nn.Identity()
        else:
            raise NotImplementedError(pos_embed_mode)

        # 0, 1 or more cls tokens
        if cls_tokens is None:
            self.cls_tokens = None
            self.static_ctx["num_cls_tokens"] = self.num_cls_tokens = 0
        else:
            self.cls_tokens = VitClassTokens(dim=dim, **cls_tokens)
            self.static_ctx["num_cls_tokens"] = self.num_cls_tokens = self.cls_tokens.num_tokens

        self.xlstm = create_block_stack(
            dim=dim,
            depth=depth,
            context_length=self.patch_embed.num_patches + self.num_cls_tokens,
            bidirectional=bidirectional,
            layerscale=layerscale,
            quaddirectional=quaddirectional,
            alternation=alternation,
            sharedirs=sharedirs,
            dropout_rate=dropout_rate,
            proj_factor=proj_factor,
            add_post_blocks_norm=add_post_blocks_norm,
        )
        # stochastic depth
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.logger.info(f"using drop_path_decay: {list_to_string(dpr)}")
        else:
            dpr = [drop_path_rate] * depth
            self.logger.info(f"drop_path_rate: {drop_path_rate}")
        for i in range(depth):
            self.xlstm.blocks[i].drop_path1.drop_prob = dpr[i]

        if disable_mask:
            for i in range(depth):
                no_mask = torch.ones_like(self.xlstm.blocks[i].xlstm.mlstm_cell.causal_mask)
                self.xlstm.blocks[i].xlstm.mlstm_cell.causal_mask = no_mask


        if mode is None:
            assert self.output_shape is None
            assert self.pooling is None
            self.head = None
            self.output_shape = (self.patch_embed.num_patches + self.num_cls_tokens, dim)
        elif mode == "classifier":
            assert self.output_shape is not None and len(self.output_shape) == 1
            if self.cls_tokens is not None:
                # use cls token as pooling
                head_in_dim = self.cls_tokens.output_shape[0]
            elif self.pooling is not None:
                pooling_input_shape = (self.patch_embed.num_patches + self.num_cls_tokens, dim)
                head_in_dim = np.prod(self.pooling.get_output_shape(pooling_input_shape))
            else:
                raise NotImplementedError
            self.head = nn.Sequential(
                nn.LayerNorm(head_in_dim, eps=eps),
                nn.Linear(head_in_dim, self.output_shape[0]),
            )
            # following MAE https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257
            nn.init.trunc_normal_(self.head[1].weight, std=2e-5)
            nn.init.zeros_(self.head[1].bias)
        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        allowed_missing_keys = []
        # interpolate for different resolution
        if self.pos_embed_mode != "none":
            old_pos_embed = state_dict["pos_embed.embed"]
            if old_pos_embed.shape != self.pos_embed.embed.shape:
                self.logger.info(f"interpolate pos_embed: {old_pos_embed.shape} -> {self.pos_embed.embed.shape}")
                state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        if self.mode == "classifier":
            allowed_missing_keys += ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias"]
        # LEGACY:
        state_dict = {k: v for k, v in state_dict.items() if "causal_mask" not in k}
        missing_keys, unexpected_keys = super().load_state_dict(state_dict=state_dict, strict=False)
        if strict:
            for allowed_missing_key in allowed_missing_keys:
                if allowed_missing_key in missing_keys:
                    missing_keys.pop(missing_keys.index(allowed_missing_key))
            assert len(missing_keys) == 0
            assert len(unexpected_keys) == 0, unexpected_keys
        return missing_keys, unexpected_keys

    def get_param_group_modifiers(self):
        modifiers = []
        if self.num_cls_tokens > 0:
            modifiers.append(WeightDecayByNameModifier(name="cls_tokens.tokens", value=0.0))
        if self.pos_embed_mode == "learnable":
            modifiers.append(WeightDecayByNameModifier(name="pos_embed.embed", value=0.0))
        return modifiers

    def forward(self, x, mask_generator=None, idx=None):
        outputs = {}

        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)

        if mask_generator is not None:
            # generate mask -> apply mask
            x, mask, ids_restore, ids_shuffle = mask_generator.get_mask(x, idx=idx)
            outputs["mask"] = mask
            outputs["ids_restore"] = ids_restore
            outputs["ids_shuffle"] = ids_shuffle
        else:
            # no mask -> flatten to 1d
            x = einops.rearrange(x, "b ... d -> b (...) d")


        # add cls token
        if self.cls_tokens is not None:
            x = self.cls_tokens(x)

        # apply blocks
        x = self.xlstm(x)

        # pool
        if self.cls_tokens is not None:
            x = self.cls_tokens.pool(x)
        elif self.pooling is not None:
            x = self.pooling(x)
        if self.head is not None:
            x = self.head(x)

        outputs["main"] = x
        return outputs

    def classify(self, *args, **kwargs):
        assert self.mode == "classifier"
        outputs = self.forward(*args, **kwargs)
        return dict(main=outputs["main"])
