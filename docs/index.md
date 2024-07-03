

[[Code](https://github.com/nx-ai/vision-lstm)] 
[[Paper](https://arxiv.org/abs/2406.04303)] 
[[Models](https://github.com/NX-AI/vision-lstm?tab=readme-ov-file#pre-trained-models)] 
[[Codebase Demo Video](https://youtu.be/80kc3hscTTg)]
[[BibTeX](https://github.com/NX-AI/vision-lstm?tab=readme-ov-file#citation)]

We introduce Vision-LSTM (ViL), an adaption of [xLSTM](https://arxiv.org/abs/2405.04517) to computer vision.
In order to adjust xLSTM (an autoregressive model) to better handle non-autoregressive inputs such as images,
we employ alternating bi-directional mLSTM blocks. Odd blocks process the image row-wise from top left to bottom right, while
even blocks process the image from bottom right to top left.


<p align="center">
<img width="100%" alt="vision_lstm_schematic" src="https://raw.githubusercontent.com/nx-ai/vision-lstm/main/docs/imgs/schematic.svg">
</p>


We pre-train ViL models on ImageNet-1K, for which we attach a linear classification head and use the 
concatenation of the first and last token as input to the classifier. Afterwards, the pre-trained model is evaluated
also on transfer classification and semantic segmentation downstream tasks.

Our new model performs favorably against heavily optimized ViT baselines such as [DeiT](https://arxiv.org/abs/2012.12877)
and [Vision-Mamba](https://arxiv.org/abs/2401.09417) (Vim) on ImageNet-1K classification, ADE20K semantic segmentation
and VTAB-1K transfer classification.


<p align="center">
<img width="60%" alt="flops_vs_performance" src="https://raw.githubusercontent.com/nx-ai/vision-lstm/main/docs/imgs/flops_vs_performance.png">
</p>


We compare against a variety of isotropic models on ImageNet-1K, where ViL performs best on the tiny and small model 
scale, outperforming transformers (DeiT), CNNs (ConvNeXt) and vision adaptions of other sequential models such as
RWKV (VRWKV) and Mamba (Vim, Mamba&reg;).
On the base model scale, ViL achieves good results but heavily optimized transformer models, that underwent multiple
cycles of hyperparameter tuning, (DeiT-III) perform best.

<p align="center">
<img width="50%" alt="results_imagenet" src="https://raw.githubusercontent.com/nx-ai/vision-lstm/main/docs/imgs/results_imagenet.png">
</p>


On ADE20K semantic segmentation, ViL also performs very well, even outperforming DeiT-III-B despite the lower 
ImageNet-1K accuracy. 

<p align="center">
<img width="50%" alt="results_ade20k" src="https://raw.githubusercontent.com/nx-ai/vision-lstm/main/docs/imgs/results_ade20k.png">
</p>


On a diverse set of 19 transfer classification tasks contained in VTAB-1K benchmark, ViL performs best on the average over all 19
datasets. Notably, ViL performs exceptionally well on the 8 structured datasets of the VTAB-1K benchmark, even 
outperforming DeiT-III-B. 

<p align="center">
<img width="50%" alt="results_vtab1k" src="https://raw.githubusercontent.com/nx-ai/vision-lstm/main/docs/imgs/results_vtab1k.png">
</p>