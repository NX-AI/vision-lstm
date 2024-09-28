import einops
import torch
from torch import nn

from .vision_lstm_util import interpolate_sincos


class VisionTransformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads,
            patch_size=16,
            input_shape=(3, 224, 224),
            mlp_hidden_dim=None,
            drop_path_rate=0.,
            drop_path_decay=True,
            num_cls_tokens=1,
            layerscale=1e-4,
            num_outputs=1000,
            mode="classifier",
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # import here so that kappamodules is not required for vision_lstm
        from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d, VitClassTokens
        from kappamodules.transformer import PrenormBlock
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.input_shape = input_shape
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.mode = mode
        self.eps = eps

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            num_channels=input_shape[0],
            resolution=input_shape[1:],
            patch_size=patch_size,
        )

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # 0, 1 or more cls tokens
        self.cls_tokens = VitClassTokens(dim=dim, num_tokens=num_cls_tokens)

        # stochastic depth
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # blocks
        self.blocks = nn.ModuleList([
            PrenormBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_ctor=nn.LayerNorm,
                drop_path=dpr[i],
                layerscale=layerscale,
                eps=eps,
            )
            for i in range(depth)
        ])
        if mode == "features":
            self.head = nn.Sequential(
                nn.LayerNorm(dim, eps=eps),
            )
        elif mode == "classifier":
            self.head = nn.Sequential(
                nn.LayerNorm(dim, eps=eps),
                nn.Linear(dim, num_outputs),
            )
        else:
            raise NotImplementedError

        self.output_shape = (self.patch_embed.num_patches + self.cls_tokens.num_tokens, dim)

    def load_state_dict(self, state_dict, strict=True):
        # interpolate pos_embed for different resolution (e.g. for fine-tuning on higher-resolution)
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        if self.mode == "features":
            state_dict.pop("head.1.weight", None)
            state_dict.pop("head.1.bias", None)
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)
        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")
        # add cls token
        x = self.cls_tokens(x)
        # apply blocks
        for blk in self.blocks:
            x = blk(x)
        # last norm
        if self.mode == "features":
            x = self.head(x)
        elif self.mode == "classifier":
            x = self.head(x[:, 0])
        else:
            raise NotImplementedError
        return x
