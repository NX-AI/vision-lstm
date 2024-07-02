import math

import einops
import torch
from kappamodules.functional.pos_embed import interpolate_sincos

from vislstm.external.models_mamba import vision_mamba
from ksuit.models import SingleModel
from ksuit.utils.param_checking import to_2tuple
from ksuit.optim.param_group_modifiers import WeightDecayByNameModifier


class VisMamba(SingleModel):
    def __init__(
            self,
            patch_size,
            dim,
            depth,
            drop_path_rate=0.0,
            drop_path_decay=True,
            mode="classifier",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.static_ctx["patch_size"] = to_2tuple(patch_size)
        self.model = vision_mamba(
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            drop_path_rate=drop_path_rate,
            drop_path_decay=drop_path_decay,
            input_shape=self.input_shape,
            mode=mode,
        )

    def get_param_group_modifiers(self):
        return [
            WeightDecayByNameModifier(name="model.cls_token", value=0.0),
            WeightDecayByNameModifier(name="model.pos_embed", value=0.0),
        ]

    def load_state_dict(self, state_dict, strict=True):
        allowed_missing_keys = []
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
        if self.model.mode == "segmentation":
            # head is segmentation head if ndim==4 -> otherwise remove head (e.g. from supervised pre-training)
            if "model.head.weight" not in state_dict or state_dict["model.head.weight"].ndim != 4:
                allowed_missing_keys += ["model.head.weight", "model.head.bias"]
                state_dict = {key: value for key, value in state_dict.items() if not key.startswith("model.head.")}
        assert strict
        missing_keys, unexpected_keys = super().load_state_dict(state_dict=state_dict, strict=False)
        for allowed_missing_key in allowed_missing_keys:
            if allowed_missing_key in missing_keys:
                missing_keys.pop(missing_keys.index(allowed_missing_key))
        assert len(missing_keys) == 0, missing_keys
        assert len(unexpected_keys) == 0, unexpected_keys
        return missing_keys, unexpected_keys

    def forward(self, x):
        return dict(main=self.model(x))

    def classify(self, x):
        return self.forward(x)

    def segment(self, x):
        assert self.model.mode == "segmentation"
        return self.forward(x)["main"]
