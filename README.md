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
- vision_lstm.py

# Get started

This code-base supports simple usage of Vision-LSTM with an "architecture-only" implementation and
also a full training pipeline.

## Architecture only
The package [vision_lstm](https://github.com/NX-AI/vision-lstm/tree/main/vision_lstm) provides a standalone
implementation in the style of [timm](https://github.com/huggingface/pytorch-image-models).

If you only need the model architecture, you can load it in a single line via torchhub or copy the
[vision_lstm](https://github.com/NX-AI/vision-lstm/tree/main/vision_lstm) into your own code-base.
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

## Full training pipeline (architecture, datasets, hyperparameters, ...)

If you want to train models with our code-base, follow the setup instructions from 
[SETUP.md](https://github.com/NX-AI/vision-lstm/tree/main/src/SETUP.md).
To start runs, follow the instructions from [RUN.md](https://github.com/NX-AI/vision-lstm/tree/main/src/RUN.md).


# Pre-trained Models

Pre-trained models on ImageNet-1K can be loaded via torchhub or directly downloaded from [here](https://ml.jku.at/research/vision_lstm/download/).

```
# pre-trained models (Table 1, left)
model = torch.hub.load("nx-ai/vision-lstm", "vil-tiny")
model = torch.hub.load("nx-ai/vision-lstm", "vil-tinyplus")
model = torch.hub.load("nx-ai/vision-lstm", "vil-small")
model = torch.hub.load("nx-ai/vision-lstm", "vil-smallplus")
model = torch.hub.load("nx-ai/vision-lstm", "vil-base")

# long-sequence fine-tuned models (Table 1, right)
model = torch.hub.load("nx-ai/vision-lstm", "vil-tinyplus-stride8")
model = torch.hub.load("nx-ai/vision-lstm", "vil-smallplus-stride8")
model = torch.hub.load("nx-ai/vision-lstm", "vil-base-stride8")

# tiny models trained for only 400 epochs (Appendix A.2)
model = torch.hub.load("nx-ai/vision-lstm", "vil-tiny-e400")
model = torch.hub.load("nx-ai/vision-lstm", "vil-tinyplus-e400")
``` 

An example of how to use these models can be found in [eval.py](https://github.com/NX-AI/vision-lstm/tree/main/eval.py)
which evaluates the models on the ImageNet-1K validation set.

Checkpoints for our reimplementation of DeiT-III-T are provided as raw checkpoint [here](https://ml.jku.at/research/vision_lstm/download/).

# Other

This code-base is an improved version of the one used for [MIM-Refiner](https://github.com/ml-jku/MIM-Refiner)
for which there exists a [demo video](https://youtu.be/80kc3hscTTg) to explain various things.


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