# vision mamba
```
conda create --prefix ~/project/env/vim python=3.10.13
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install kappamodules
pip install kappaschedules
pip install kappaprofiler
pip install kappaconfig
pip install wandb
pip install torchmetrics
pip install kappadata

git clone git@github.com:hustvl/Vim.git
cd Vim
pip install -r vim/vim_requirements.txt

# make sure you are on a compute node with a GPU and that CUDA_HOME is set
# lux: ml load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.2.0
cd causal-conv1d
# if "error: [Errno 18] Invalid cross-device link: ..." -> change https://github.com/hustvl/Vim/blob/main/causal-conv1d/setup.py#L219 to shutil.copy
pip install .

# change "causal_conv1d>=1.1.0" to "causal_conv1d==1.1.0" in setup.py  
# (https://github.com/hustvl/Vim/issues/34#issuecomment-1977102845)
cd mamba-1p1p1
pip install .

# i had to do this because for some reason the original mamba was installed
mv /home/users/u100316/project/env/vim/lib/python3.10/site-packages/mamba_ssm /home/users/u100316/project/env/vim/lib/python3.10/site-packages/mamba_ssm_bkp
cp -R ~/Vim/mamba-1p1p1/mamba_ssm /home/users/u100316/project/env/vim/lib/python3.10/site-packages/
```

Might have to do the following if Mamba got unexpected "bitype": `cp ~/Vim/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py /home/users/u100316/project/env/vim/lib/python3.10/site-packages/mamba_ssm/modules/`

## check vision mamba installation
```
cd vim
from models_mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
m = vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2().to("cuda")
import torch
x = torch.randn(1, 3, 224, 224).to("cuda")
m(x).shape
```


# mamba (old)
- change https://github.com/hustvl/Vim/blob/main/causal-conv1d/setup.py#L219 to shutil.copy
- follow setup from https://github.com/hustvl/Vim?tab=readme-ov-file#envs-for-pretraining

other requirements
- pip install kappamodules
- pip install kappaschedules
- pip install kappaprofiler
- pip install kappaconfig
- pip install wandb
- pip install torchmetrics
requirements that will be removed in the future
- pip install kappadata