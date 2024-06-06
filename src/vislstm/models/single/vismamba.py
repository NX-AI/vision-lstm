import math

import einops
import torch
from kappamodules.functional.pos_embed import interpolate_sincos

from vislstm.external.models_mamba import vision_mamba
from ksuit.models import SingleModel
from ksuit.optim.param_group_modifiers import WeightDecayByNameModifier


class VisMamba(SingleModel):
    def __init__(
            self,
            patch_size,
            dim,
            depth,
            drop_path_rate=0.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model = vision_mamba(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            drop_path_rate=drop_path_rate,
            input_shape=self.input_shape,
        )

    def get_param_group_modifiers(self):
        return [
            WeightDecayByNameModifier(name="model.cls_token", value=0.0),
            WeightDecayByNameModifier(name="model.pos_embed", value=0.0),
        ]

    def load_state_dict(self, state_dict, strict=True):
        # interpolate pos_embed
        pos_embed = state_dict["model.pos_embed"]
        if pos_embed.shape != self.model.pos_embed.shape:
            self.logger.info(f"interpolate pos_embed: {pos_embed.shape} -> {self.model.pos_embed.shape}")
            cls = pos_embed[:, :1]
            pos_embed = pos_embed[:, 1:]
            assert math.sqrt(pos_embed.size(1)).is_integer()
            h = w = int(math.sqrt(pos_embed.size(1)))
            pos_embed = einops.rearrange(pos_embed, "1 (h w) c -> 1 h w c", h=h, w=w)
            seqlen = int(math.sqrt(self.model.pos_embed.size(1) - 1))
            pos_embed = interpolate_sincos(embed=pos_embed, seqlens=[seqlen, seqlen])
            pos_embed = einops.rearrange(pos_embed, "1 h w c -> 1 (h w) c")
            pos_embed = torch.concat([cls, pos_embed], dim=1)
            state_dict["model.pos_embed"] = pos_embed
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, x):
        return dict(main=self.model(x))

    def classify(self, x):
        return self.forward(x)
