# Vision-LSTM (ViL)

[[`Project Page`](https://nx-ai.github.io/vision-lstm)] 
[[`Paper`](https://arxiv.org/abs/2406.04303)] 
[[`Models`](https://github.com/nx-ai/vision-lstm#pre-trained-models)] 
[[`Codebase Demo Video`](https://youtu.be/80kc3hscTTg)]
[[`BibTeX`](https://github.com/nx-ai/vision-lstm#citation)]

Pytorch implementation and pre-trained models of Vision-LSTM (ViL), an adaption of xLSTM to computer vision.

<p align="center">
<img width="100%" alt="vision_lstm_schematic" src="https://raw.githubusercontent.com/nx-ai/vision-lstm/main/docs/imgs/schematic.svg">
</p>

## License

This project is licensed under the [MIT](https://github.com/NX-AI/vision-lstm?tab=MIT-2-ov-file) License, except the following folders/files, 
which are licensed under the [AGPL-3.0](https://github.com/NX-AI/vision-lstm?tab=AGPL-3.0-1-ov-file) license:
- src/vislstm/modules/xlstm
- vision_lstm/vision_lstm.py
- vision_lstm/vision_lstm2.py

# Get started

This code-base supports simple usage of Vision-LSTM with an "architecture-only" implementation and
also a full training pipeline.

## Architecture only
The package [vision_lstm](https://github.com/NX-AI/vision-lstm/tree/main/vision_lstm) provides a standalone
implementation in the style of [timm](https://github.com/huggingface/pytorch-image-models).

If you only need the model architecture, you can load it in a single line via torchhub or copy the
[vision_lstm](https://github.com/NX-AI/vision-lstm/tree/main/vision_lstm) folder into your own code-base.
Note that for `VisionLSTM2` we consider a single block to consist of two subblocks (the first one going from top-right 
to bottom-left and the second one going from bottom-right to top-left) to ease implementations of 
layerwise learning rate decay.
```
# load ViL-T
model = torch.hub.load("nx-ai/vision-lstm", "VisionLSTM2")
# load your own model
model = torch.hub.load(
    "nx-ai/vision-lstm", 
    "VisionLSTM2",  # VisionLSTM2 is an improved version over VisionLSTM
    dim=192,  # latent dimension (192 for ViL-T)
    depth=12,  # how many ViL blocks (1 block consists 2 subblocks of a forward and backward block)
    patch_size=16,  # patch_size (results in 196 patches for 224x224 images)
    input_shape=(3, 224, 224),  # RGB images with resolution 224x224
    output_shape=(1000,),  # classifier with 1000 classes
    drop_path_rate=0.05,  # stochastic depth parameter
)
```

See [below](https://github.com/NX-AI/vision-lstm?tab=readme-ov-file#version1-pre-trained-models) or 
[Appendix A](https://arxiv.org/abs/2406.04303)) for a list of changes between `VisionLSTM` and `VisionLSTM2`. 
We recommend to use `VisionLSTM2` as we found it to perform better but keep `VisionLSTM` for backward compatibility.

## Full training pipeline (architecture, datasets, hyperparameters, ...)

If you want to train models with our code-base, follow the setup instructions from 
[SETUP.md](https://github.com/NX-AI/vision-lstm/tree/main/src/SETUP.md).
To start runs, follow the instructions from [RUN.md](https://github.com/NX-AI/vision-lstm/tree/main/src/RUN.md).


# Pre-trained models

Pre-trained models on ImageNet-1K can be loaded via torchhub or directly downloaded from [here](https://ml.jku.at/research/vision_lstm/download/).

```
# ImageNet-1K pre-trained models
model = torch.hub.load("nx-ai/vision-lstm", "vil2-tiny")               # 78.3%
model = torch.hub.load("nx-ai/vision-lstm", "vil2-small")              # 81.5%
model = torch.hub.load("nx-ai/vision-lstm", "vil2-base")               # 82.4%

# ViL-T trained for only 400 epochs (Appendix B.2)
model = torch.hub.load("nx-ai/vision-lstm", "vil2-tiny-e400")          # 77.2%
``` 

Pre-training logs of these models can be found [here](https://github.com/NX-AI/vision-lstm/tree/main/logs/pretrain).

An example of how to use these models can be found in [eval.py](https://github.com/NX-AI/vision-lstm/tree/main/eval.py)
which evaluates the models on the ImageNet-1K validation set.


## DeiT-III-T reimplementation models
Checkpoints for our reimplementation of DeiT-III-T are provided as raw checkpoint 
[here](https://ml.jku.at/research/vision_lstm/download/) and can be loaded from torchhub 
(the vision transformer implementation is based on [KappaModules](https://github.com/BenediktAlkin/KappaModules) so 
you need to install it before loading a ViT checkpoint via torchhub by running `pip install kappamodules==0.1.70`).

```
model = torch.hub.load("nx-ai/vision-lstm", "deit3-tiny-e400")  # 75.6%
model = torch.hub.load("nx-ai/vision-lstm", "deit3-tiny")       # 76.2%
```

# Version1 pre-trained models

In the first iteration of ViL, models were trained with (i) bilateral_avg pooling instead of bilateral_concat 
(ii) causal conv1d instead of conv2d before q and k (iii) no biases in projection and layernorms (iv) 224 resolution
for the whole training process instead of pre-training at 192 resolution followed by a short fine-tuning on 224 
resolution. These changes improve ImageNet-1K accuracy of a ViL-T from 77.3% to 78.3%. See Appendix A in the paper
for more details. We recommend to use VisionLSTM2 instead of VisionLSTM but keep support for the initial version as-is.
Pre-trained models of the first iteration can be loaded as follows:

```
# ImageNet-1K pre-trained models
model = torch.hub.load("nx-ai/vision-lstm", "vil-tiny")               # 77.3%
model = torch.hub.load("nx-ai/vision-lstm", "vil-tinyplus")           # 78.1%
model = torch.hub.load("nx-ai/vision-lstm", "vil-small")              # 80.7%
model = torch.hub.load("nx-ai/vision-lstm", "vil-smallplus")          # 80.9%
model = torch.hub.load("nx-ai/vision-lstm", "vil-base")               # 81.6%

# long-sequence fine-tuned models
model = torch.hub.load("nx-ai/vision-lstm", "vil-tinyplus-stride8")   # 80.0%
model = torch.hub.load("nx-ai/vision-lstm", "vil-smallplus-stride8")  # 82.2%
model = torch.hub.load("nx-ai/vision-lstm", "vil-base-stride8")       # 82.7%

# tiny models trained for only 400 epochs
model = torch.hub.load("nx-ai/vision-lstm", "vil-tiny-e400")          # 76.1%
model = torch.hub.load("nx-ai/vision-lstm", "vil-tinyplus-e400")      # 77.2%
``` 

Initializing with random weights can be done as follows:

```
# load ViL-T
model = torch.hub.load("nx-ai/vision-lstm", "VisionLSTM")
# load your own model
model = torch.hub.load(
    "nx-ai/vision-lstm", 
    "VisionLSTM",
    dim=192,  # latent dimension (192 for ViL-T)
    depth=24,  # how many ViL blocks
    patch_size=16,  # patch_size (results in 196 patches for 224x224 images)
    input_shape=(3, 224, 224),  # RGB images with resolution 224x224
    output_shape=(1000,),  # classifier with 1000 classes
    drop_path_rate=0.05,  # stochastic depth parameter
    stride=None,  # set to 8 for long-sequence fine-tuning
)
```

# Other

This code-base is an improved version of the one used for [MIM-Refiner](https://github.com/ml-jku/MIM-Refiner)
for which there exists a [demo video](https://youtu.be/80kc3hscTTg) to explain various things.


VTAB-1K evaluations were conducted with [this](https://github.com/BenediktAlkin/vtab1k-pytorch) codebase. 

# Citation

If you like our work, please consider giving it a star :star: and cite us

```
@article{alkin2024visionlstm,
  title={Vision-LSTM: xLSTM as Generic Vision Backbone},
  author={Benedikt Alkin and Maximilian Beck and Korbinian P{\"o}ppel and Sepp Hochreiter and Johannes Brandstetter}
  journal={arXiv preprint arXiv:2406.04303},
  year={2024}
}
```