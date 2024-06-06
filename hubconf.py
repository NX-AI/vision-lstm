from functools import partial
from vision_lstm import VisionLSTM
import torch

dependencies = ["torch", "einops"]

CONFIGS = {
    # tiny
    "vil-tiny": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=24, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tiny16_e800_in1k.th",
        preprocess="v1",
    ),
    "vil-tiny-e400": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=24, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tiny16_e400_in1k.th",
        preprocess="v1",
    ),
    "vil-tinyplus": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=29, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tinyplus16_e800_in1k.th",
        preprocess="v1",
    ),
    "vil-tinyplus-e400": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=29, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tinyplus16_e400_in1k.th",
        preprocess="v1",
    ),
    # tiny-longseq
    "vil-tinyplus-stride8": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=29, stride=8, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tinyplus16_e800_in1k.th",
        preprocess="v1",
    ),
    # small
    "vil-small": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=384, depth=24, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_small16_e400_in1k.th",
        preprocess="v1",
    ),
    "vil-smallplus": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=384, depth=26, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_smallplus16_e400_in1k.th",
        preprocess="v1",
    ),
    # small-longseq
    "vil-smallplus-stride8": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=384, depth=26, stride=8, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_smallplus16_e400_in1k_stride8.th",
        preprocess="v1",
    ),
    # base
    "vil-base": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=768, depth=24, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_base16_e400_in1k.th",
        preprocess="v1",
    ),
    "vil-base-stride8": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=768, depth=24, stride=8, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_base16_e400_in1k_stride8.th",
        preprocess="v1",
    ),
}


def load_model(ctor, ctor_kwargs, url=None, pretrained=True, preprocess=None, **kwargs):
    model = ctor(**ctor_kwargs, **kwargs)
    if pretrained:
        assert url is not None, f"pretrained=True but no url found"
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        sd = sd["state_dict"]
        if preprocess is None:
            pass
        elif preprocess == "v1":
            sd = {key.replace(".xlstm.", ".layer."): value for key, value in sd.items()}
            sd = {key.replace("xlstm.", ""): value for key, value in sd.items()}
            sd = {key.replace(".xlstm_norm.", ".norm."): value for key, value in sd.items()}
            sd["legacy_norm.weight"] = sd.pop("post_blocks_norm.weight")
            sd["norm.weight"] = sd.pop("head.0.weight")
            sd["norm.bias"] = sd.pop("head.0.bias")
            sd["head.weight"] = sd.pop("head.1.weight")
            sd["head.bias"] = sd.pop("head.1.bias")
        else:
            raise NotImplementedError(f"invalid checkpoint preprocessing '{preprocess}'")
        model.load_state_dict(sd)
    return model


for name, config in CONFIGS.items():
    globals()[name] = partial(load_model, **config)
