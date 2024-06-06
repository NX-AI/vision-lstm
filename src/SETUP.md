# Setup python environment

Create a python environment either via the provided 
[environment.yml](https://github.com/NX-AI/vision-lstm/tree/main/src/environment.yml) file 
(e.g. via `conda env create -n vil -f src/environment.yml`). You probably need to adjust the installed 
torch/torchvision versions as installing via the `environment.yml` sometimes installs the cpu-only version or a wrong 
cuda version.

```
pip install torch==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.16.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```


You can also manually install the dependencies via pip  

```
pip install torch==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.16.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install kappamodules==0.1.70
pip install kappaschedules==0.0.31
pip install kappadata==1.4.13
pip install kappaconfig==1.0.31
pip install kappaprofiler==1.0.11
pip install wandb
pip install einops
pip install torchmetrics
```

# static_config

This codebase relies on a `static_config.yaml` file which defines static properties about your compute environment
(e.g. path to datasets, paths where to store checkpoints/logs, ...). A template is provided 
[here](https://github.com/NX-AI/vision-lstm/tree/main/src/setup/static_configs/prod.yaml).

Simply copy the template and adjust the values to your setup.
`cp src/setup/static_configs/prod.yaml src/static_config.yaml`



# OPTIONAL: log to Weights and Biases

If you want to log metrics to [Weights & Biases](https://wandb.ai/), you need to specify the entity and project in
a `wandb_config.yaml` file. A template is provided 
[here](https://github.com/NX-AI/vision-lstm/tree/main/src/setup/wandb_configs/official.yaml).

Simply copy the template and adjust the values to your setup.
`cp src/setup/wandb_configs/official.yaml src/wandb_config.yaml`

NOTE: you will be asked for your API key on your first wandb login

If you don't want to log things to wandb, set the default_wandb_mode in the `static_config.yaml` to `disabled`.


# OPTIONAL: run via SLURM

You can start runs on a SLURM managed cluster via `python main_sbatch.py --time TIME --hp HP --nodes NODES`.
For examples: `python main_sbatch.py --time 48:00:00 --hp vislstm/yamls/vil/pretrain/lstm_90M16_e400_bialter_bilatavg_lr1e3_sd02unif_res224.yaml --nodes 4`
will pretrain a ViL-B on 4 compute nodes.

In order to do that, you to setup a `sbatch_config.yaml`, for which a template is provided in `src/setup/sbatch_configs/prod.yaml`
and a `sbatch_template.sh` for which a template is provided in `src/setup/sbatch_templates/prod_nodes.sh`.
The instructions here only provide a setup to use full nodes (no partial nodes), hence the postfix `_nodes`.

- `cp src/setup/sbatch_configs/prod.yaml sbatch_config.yaml`
- `src/setup/sbatch_templates/prod_nodes.sh sbatch_template_nodes.sh`
- adjust values to setup



# OPTIONAL: log runs to different wandb projects

You can log different runs to different wandb projects.

```
# create folder that contains all your wandb projects that you want to log to 
mkdir wandb_configs
# create a wandb_config for each project
cp setup/wandb_configs/official.yaml wandb_configs/<NAME>.yaml
# adjust values to your setup
nano wandb_configs/<NAME>.yaml
```

Specify the `<NAME>` of the wandb config either with  `python main_train.py ... --wandb_config <NAME>` or directly
in the yaml file that you pass to `--hp`
```
wandb: <NAME>
datasets: ...
model: ...
trainer: ...
```

# Troubleshooting

The code-base is an improved version of the one used for [MIM-Refiner](https://github.com/ml-jku/MIM-Refiner)
for which there exists a [demo video](https://youtu.be/80kc3hscTTg) which includes parts of this setup.
So this video might solve your issue with setting up the codebase.
