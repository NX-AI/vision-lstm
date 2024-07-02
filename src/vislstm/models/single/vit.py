from functools import partial

import einops
import torch
import torch.nn.functional as F
from kappamodules.attention import DotProductAttention1d
from kappamodules.functional.pos_embed import interpolate_sincos
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.transformer import PrenormBlock
from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d, VitClassTokens
from torch import nn

from ksuit.factory import MasterFactory
from ksuit.models import SingleModel
from ksuit.models.poolings import ToImage
from ksuit.optim.param_group_modifiers import WeightDecayByNameModifier
from ksuit.utils.formatting_utils import list_to_string
from ksuit.utils.param_checking import to_ntuple


class Vit(SingleModel):
    def __init__(
            self,
            patch_size,
            dim,
            depth,
            num_attn_heads,
            stride=None,
            mlp_hidden_dim=None,
            drop_path_rate=0.,
            drop_path_decay=False,
            num_cls_tokens=1,
            layerscale=None,
            learnable_pos_embed=False,
            init_weights="truncnormal002",
            flash_attention=True,
            mode="features",
            pooling=None,
            use_relpos_bias=False,
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        ndim = len(self.input_shape) - 1
        self.patch_size = to_ntuple(patch_size, n=ndim)
        self.static_ctx["patch_size"] = self.patch_size
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.mode = mode
        self.pooling = MasterFactory.get("pooling").create(pooling, static_ctx=self.static_ctx)
        self.eps = eps
        self.use_relpos_bias = use_relpos_bias

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
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim, is_learnable=learnable_pos_embed)
        self.logger.info(f"pos_embed.is_learnable={self.pos_embed.is_learnable}")

        # 0, 1 or more cls tokens
        self.cls_tokens = VitClassTokens(dim=dim, num_tokens=num_cls_tokens)
        self.static_ctx["num_cls_tokens"] = self.num_cls_tokens = num_cls_tokens

        # stochastic depth
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.logger.info(f"using drop_path_decay: {list_to_string(dpr)}")
        else:
            dpr = [drop_path_rate] * depth
            self.logger.info(f"drop_path_rate: {drop_path_rate}")

        # option to disable flashattn for benchmarking
        block_kwargs = {}
        if not flash_attention:
            from kappamodules.attention import DotProductAttentionSlow
            block_kwargs["attn_ctor"] = DotProductAttentionSlow

        # blocks
        if use_relpos_bias:
            block_kwargs["attn_ctor"] = partial(
                DotProductAttention1d,
                rel_pos_bias="learnable",
                seqlens=self.patch_embed.seqlens,
            )
        self.blocks = nn.ModuleList([
            PrenormBlock(
                dim=dim,
                num_heads=num_attn_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_ctor=nn.LayerNorm,
                drop_path=dpr[i],
                layerscale=layerscale,
                eps=eps,
                init_weights=init_weights,
                **block_kwargs,
            )
            for i in range(self.depth)
        ])

        if mode == "features":
            assert self.output_shape is None
            self.head = None
            self.output_shape = (self.patch_embed.num_patches + self.num_cls_tokens, dim)
            if self.pooling is not None:
                self.output_shape = self.pooling.get_output_shape(self.output_shape)
        elif mode == "segmentation":
            assert self.output_shape is not None and len(self.output_shape) == 3
            assert self.pooling is None
            self.pooling = ToImage(static_ctx=self.static_ctx)
            self.head = nn.Conv2d(dim, self.output_shape[0], kernel_size=1)
            init_xavier_uniform_zero_bias(self.head)
        elif mode == "classifier":
            assert self.output_shape is not None and len(self.output_shape) == 1
            assert self.pooling is not None
            head_in_dim = self.pooling.get_output_shape((self.patch_embed.num_patches + self.num_cls_tokens, dim))[0]
            self.head = nn.Sequential(
                nn.LayerNorm(dim, eps=eps),
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
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            self.logger.info(f"interpolate pos_embed: {old_pos_embed.shape} -> {self.pos_embed.embed.shape}")
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        # rel_pos is added after pretraining
        if self.use_relpos_bias:
            allowed_missing_keys += [key for key in self.state_dict().keys() if ".rel_pos_" in key]
        if self.mode == "features":
            # remove head (e.g. supervised pre-training -> UperNet semantic segmentation)
            for key in list(state_dict.keys()):
                if key.startswith("head."):
                    state_dict.pop(key)
        if self.mode == "classifier":
            allowed_missing_keys += ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias"]
            # reinitialize head for transfer classification
            if "head.1.bias" in state_dict and state_dict["head.1.bias"].shape != self.head[1].bias.shape:
                self.logger.info(f"classification head shape doesnt match -> reinitialize classification head")
                state_dict.pop("head.1.weight")
                state_dict.pop("head.1.bias")
        if self.mode == "segmentation":
            # head is segmentation head if ndim==4 -> otherwise remove head (e.g. from supervised pre-training)
            if "head.weight" not in state_dict or state_dict["head.weight"].ndim != 4:
                allowed_missing_keys += ["head.weight", "head.bias"]
                state_dict = {key: value for key, value in state_dict.items() if not key.startswith("head.")}
        missing_keys, unexpected_keys = super().load_state_dict(state_dict=state_dict, strict=False)
        if strict:
            for allowed_missing_key in allowed_missing_keys:
                if allowed_missing_key in missing_keys:
                    missing_keys.pop(missing_keys.index(allowed_missing_key))
            assert len(missing_keys) == 0, missing_keys
            assert len(unexpected_keys) == 0, unexpected_keys
        return missing_keys, unexpected_keys

    def get_param_group_modifiers(self):
        modifiers = []
        if self.cls_tokens.num_tokens > 0:
            modifiers.append(WeightDecayByNameModifier(name="cls_tokens.tokens", value=0.0))
        if self.pos_embed.is_learnable:
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
        x = self.cls_tokens(x)

        # apply blocks
        for blk in self.blocks:
            x = blk(x)

        if self.pooling is not None:
            x = self.pooling(x)
        if self.head is not None:
            x = self.head(x)

        if self.mode == "segmentation":
            # interpolate is not supported in bfloat16
            with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
                x = F.interpolate(x.float(), size=self.output_shape[1:], mode="bilinear", align_corners=False)

        outputs["main"] = x
        return outputs

    def classify(self, *args, **kwargs):
        assert self.mode == "classifier"
        outputs = self.forward(*args, **kwargs)
        return dict(main=outputs["main"])

    def segment(self, x):
        assert self.mode == "segmentation"
        return self.forward(x)["main"]
