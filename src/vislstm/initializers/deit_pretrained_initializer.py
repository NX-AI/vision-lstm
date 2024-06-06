from pathlib import Path

import einops
import torch

from ksuit.initializers.base import InitializerBase
from ksuit.distributed import is_rank0, barrier

class DeitPretrainedInitializer(InitializerBase):
    """ initialize with weights from a state_dict loaded via torchhub """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def _get_model_kwargs(self):
        model = self.model.lower().replace("_", "")
        if "small" in model:
            return dict(patch_size=16, dim=384, num_attn_heads=6, depth=12)
        if "base" in model:
            return dict(
                patch_size=16,
                dim=768,
                num_attn_heads=12,
                depth=12,
                drop_path_rate=0.1,
                drop_path_decay=False,
            )
        raise NotImplementedError(f"get_model_kwargs of '{self.model}' is not implemented")

    def init_weights(self, model):
        self.logger.info(f"loading DeiT weights of model '{self.model}'")
        if self.model == "small_res224_in1k":
            url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        elif self.model == "base_res224_in1k":
            url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
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
        sd["pos_embed.embed"] = pos_embed[:, 1:].reshape(*model.pos_embed.embed.shape)
        # kappamodules has different key for CLS token + no pos_embed for CLS
        sd["cls_tokens.tokens"] = sd.pop("cls_token") + pos_embed[:, :1]
        # norm + head is merged into sequential
        sd["head.0.weight"] = sd.pop("norm.weight")
        sd["head.0.bias"] = sd.pop("norm.bias")
        sd["head.1.weight"] = sd.pop("head.weight")
        sd["head.1.bias"] = sd.pop("head.bias")
        model.load_state_dict(sd)
