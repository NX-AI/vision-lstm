

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


We train and evaluate ViL models on ImageNet-1K, for which we attach a linear classification head and use the average
of the first and last token as input to the classifier.

Our new model can outperform heavily optimized ViT baselines such as [DeiT](https://arxiv.org/abs/2012.12877) on 
smaller models and also heavily outperforms other sequential vision models such as 
[Vision-Mamba](https://arxiv.org/abs/2401.09417) (Vim)

<p align="center">
<img width="40%" alt="results_tiny_small" src="https://raw.githubusercontent.com/nx-ai/vision-lstm/main/docs/imgs/results_tiny_small.png">
</p>


However, as ViT training pipelines were heavily optimized over the last few years, ViL doesn't outperform the most 
optimized ViT baselines yet. Nevertheless, it shows strong performances by achieving comparable results to 
[DeiT](https://arxiv.org/abs/2012.12877). Note that training on this scale is still quite costly due to lack of custom
hardware implementations such as CUDA kernels. So the hyperparameters for this scale are far from optimal.

<p align="center">
<img width="40%" alt="results_base" src="https://raw.githubusercontent.com/nx-ai/vision-lstm/main/docs/imgs/results_base.png">
</p>