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
    ),
    "vil-tiny-e400": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=24, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tiny16_e400_in1k.th",
    ),
    "vil-tinyplus": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=29, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tinyplus16_e800_in1k.th",
    ),
    "vil-tinyplus-e400": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=29, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tinyplus16_e400_in1k.th",
    ),
    # tiny-longseq
    "vil-tinyplus-stride8": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=192, depth=29, stride=8, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_tinyplus16_e800_in1k.th",
    ),
    # small
    "vil-small": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=384, depth=24, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_small16_e400_in1k.th",
    ),
    "vil-smallplus": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=384, depth=26, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_smallplus16_e400_in1k.th",
    ),
    # small-longseq
    "vil-smallplus-stride8": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=384, depth=26, stride=8, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_smallplus16_e400_in1k_stride8.th",
    ),
    # base
    "vil-base": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=768, depth=24, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_base16_e400_in1k.th",
    ),
    "vil-base-stride8": dict(
        ctor=VisionLSTM,
        ctor_kwargs=dict(dim=768, depth=24, stride=8, legacy_norm=True),
        url="https://ml.jku.at/research/vision_lstm/download/vil_base16_e400_in1k_stride8.th",
    ),
}


def load_model(ctor, ctor_kwargs, url=None, pretrained=True, **kwargs):
    model = ctor(**ctor_kwargs, **kwargs)
    if pretrained:
        assert url is not None, f"pretrained=True but no url found"
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(sd["state_dict"])
    return model


for name, config in CONFIS.items():
    globals()[name] = partial(load_model, **config)
