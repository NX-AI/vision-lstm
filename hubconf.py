from functools import partial

import torch

from vision_lstm import VisionLSTM, VisionLSTM2, VisionTransformer

dependencies = ["torch", "einops"]

CONFIGS_VIT = {
    # deit3-reimpl
    "deit3-tiny-e400": dict(
        ctor=VisionTransformer,
        ctor_kwargs=dict(patch_size=16, dim=192, depth=12, num_heads=3),
        url="https://ml.jku.at/research/vision_lstm/download/vit_tiny16_e400_in1k_deit3reimpl.th",
    ),
    "deit3-tiny": dict(
        ctor=VisionTransformer,
        ctor_kwargs=dict(patch_size=16, dim=192, depth=12, num_heads=3),
        url="https://ml.jku.at/research/vision_lstm/download/vit_tiny16_e800_in1k_deit3reimpl.th",
    ),
}
CONFIGS_V1 = {
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
CONFIGS_V2 = {
    # tiny
    "vil2-tiny": dict(
        ctor=VisionLSTM2,
        ctor_kwargs=dict(
            dim=192,
            depth=12,
            legacy_norm=True,
            pooling="bilateral_flatten",
            conv_kind="2d",
            conv_kernel_size=3,
            norm_bias=True,
            proj_bias=True,
        ),
        url="https://ml.jku.at/research/vision_lstm/download/vil2_tiny16_e800_in1k.th",
        preprocess="v2",
    ),
    "vil2-tiny-e400": dict(
        ctor=VisionLSTM2,
        ctor_kwargs=dict(
            dim=192,
            depth=12,
            legacy_norm=True,
            pooling="bilateral_flatten",
            conv_kind="2d",
            conv_kernel_size=3,
            norm_bias=True,
            proj_bias=True,
        ),
        url="https://ml.jku.at/research/vision_lstm/download/vil2_tiny16_e400_in1k.th",
        preprocess="v2",
    ),
    # small
    "vil2-small": dict(
        ctor=VisionLSTM2,
        ctor_kwargs=dict(
            dim=384,
            depth=12,
            legacy_norm=True,
            pooling="bilateral_flatten",
            conv_kind="2d",
            conv_kernel_size=3,
            norm_bias=True,
            proj_bias=True,
        ),
        url="https://ml.jku.at/research/vision_lstm/download/vil2_small16_e400_in1k.th",
        preprocess="v2",
    ),
    # base
    "vil2-base": dict(
        ctor=VisionLSTM2,
        ctor_kwargs=dict(
            dim=768,
            depth=12,
            legacy_norm=True,
            pooling="bilateral_flatten",
            conv_kind="2d",
            conv_kernel_size=3,
            norm_bias=True,
            proj_bias=True,
        ),
        url="https://ml.jku.at/research/vision_lstm/download/vil2_base16_e400_in1k.th",
        preprocess="v2",
    ),
}


def load_model(ctor, ctor_kwargs, url=None, pretrained=True, preprocess=None, **kwargs):
    model = ctor(**ctor_kwargs, **kwargs)
    if pretrained:
        assert url is not None, f"pretrained=True but no url found"
        if url.startswith("https://"):
            sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        else:
            # load from disk for debugging
            from pathlib import Path
            sd = torch.load(Path(url).expanduser(), map_location="cpu")
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
        elif preprocess == "v2":
            sd = {key.replace(".xlstm.", ".layer."): value for key, value in sd.items()}
            sd = {key.replace("xlstm.", ""): value for key, value in sd.items()}
            sd = {key.replace(".xlstm_norm.", ".norm."): value for key, value in sd.items()}
            sd = {key.replace(".conv1d.", ".conv."): value for key, value in sd.items()}
            depth = ctor_kwargs["depth"]
            for i in range(depth * 2):
                if i % 2 == 0:
                    sd = {
                        key.replace(f"blocks.{i}.", f"blocks.{i // 2}.rowwise_from_top_left."): value
                        for key, value in sd.items()
                    }
                else:
                    sd = {
                        key.replace(f"blocks.{i}.", f"blocks.{i // 2}.rowwise_from_bot_right."): value
                        for key, value in sd.items()
                    }
            sd["norm.weight"] = sd.pop("post_blocks_norm.weight")
            sd["norm.bias"] = sd.pop("post_blocks_norm.bias")
            if ctor_kwargs["legacy_norm"]:
                sd["legacy_norm.weight"] = sd.pop("head.0.weight")
                sd["legacy_norm.bias"] = sd.pop("head.0.bias")
            sd["head.weight"] = sd.pop("head.1.weight")
            sd["head.bias"] = sd.pop("head.1.bias")
        else:
            raise NotImplementedError(f"invalid checkpoint preprocessing '{preprocess}'")
        model.load_state_dict(sd)
    return model


for name, config in CONFIGS_VIT.items():
    globals()[name] = partial(load_model, **config)
for name, config in CONFIGS_V1.items():
    globals()[name] = partial(load_model, **config)
for name, config in CONFIGS_V2.items():
    globals()[name] = partial(load_model, **config)
