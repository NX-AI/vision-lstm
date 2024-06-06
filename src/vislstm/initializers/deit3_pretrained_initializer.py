from pathlib import Path

import einops
import torch

from ksuit.initializers.base import InitializerBase
from ksuit.distributed import is_rank0, barrier

class Deit3PretrainedInitializer(InitializerBase):
    """ initialize with weights from a state_dict loaded via torchhub """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def _get_model_kwargs(self):
        model = self.model.lower().replace("_", "")
        if "small" in model:
            return dict(
                patch_size=16,
                dim=384,
                num_attn_heads=6,
                depth=12,
                layerscale=1e-4,
                drop_path_rate=0.05,
                drop_path_decay=False,
            )
        if "base" in model:
            return dict(
                patch_size=16,
                dim=768,
                num_attn_heads=12,
                depth=12,
                layerscale=1e-4,
                drop_path_rate=0.2,
                drop_path_decay=False,
            )
        if "large" in model:
            return dict(
                patch_size=16,
                dim=1024,
                num_attn_heads=16,
                depth=24,
                layerscale=1e-4,
                drop_path_rate=0.45,
                drop_path_decay=False,
            )
        raise NotImplementedError(f"get_model_kwargs of '{self.model}' is not implemented")

    def init_weights(self, model):
        self.logger.info(f"loading DeiT-III weights of model '{self.model}'")
        if self.model == "small_res224_in1k":
            url = "https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth"
        elif self.model == "base_res224_in1k":
            url = "https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth"
        elif self.model == "large_res224_in1k":
            url = "https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth"
        else:
            raise NotImplementedError
        if is_rank0():
            sd = torch.hub.load_state_dict_from_url(url)
            barrier()
        else:
            # wait for rank0 to download
            barrier()
            # load cached model
            sd = torch.hub.load_state_dict_from_url(url)
        sd = sd["model"]
        assert "pos_embed" in sd
        pos_embed = sd.pop("pos_embed")
        sd["pos_embed.embed"] = pos_embed.reshape(*model.pos_embed.embed.shape)
        # kappamodules has different key for CLS token
        sd["cls_tokens.tokens"] = sd.pop("cls_token")
        # norm + head is merged into sequential
        sd["head.0.weight"] = sd.pop("norm.weight")
        sd["head.0.bias"] = sd.pop("norm.bias")
        sd["head.1.weight"] = sd.pop("head.weight")
        sd["head.1.bias"] = sd.pop("head.bias")
        # layerscale is different in
        for key in list(sd.keys()):
            if key.endswith(".gamma_1"):
                sd[key.replace(".gamma_1", ".ls1.gamma")] = sd.pop(key)
            if key.endswith(".gamma_2"):
                sd[key.replace(".gamma_2", ".ls2.gamma")] = sd.pop(key)
        model.load_state_dict(sd)
