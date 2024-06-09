# Start a single run

`main_train.py` starts a training run. You need to define the number of GPUs (via `--devices`) and the hyperparameters
(via `--hp`) for a run. For additional options, see `python main_train.py --help`. 

```
# train on a single GPU with index 0
python main_train.py --devices 0 --hp vislstm/yamls/vil/pretrain/lstm_6M16_e400_bialter_bilatavg_lr1e3_res224.yaml
# train on 4 GPUs
python main_train.py --devices 0,1,2,3 --hp vislstm/yamls/vil/pretrain/lstm_6M16_e400_bialter_bilatavg_lr1e3_res224.yaml
# train on 4 specific GPUs
python main_train.py --devices 0,2,5,6 --hp vislstm/yamls/vil/pretrain/lstm_6M16_e400_bialter_bilatavg_lr1e3_res224.yaml

# debug run: batchsize, dataset size and number of epochs will be heavily reduced
python main_train.py --testrun --devices 0 --hp vislstm/yamls/vil/pretrain/lstm_6M16_e400_bialter_bilatavg_lr1e3_res224.yaml
```


# Queue up runs

`main_run_folder.py` provides a simple queueing system. Simply copy the yamls that you want to run into a folder 
and start the runs. By default the directory `yamls_run` will be used (which can be changed via `--folder PATH`)

```
# run all yamls from the yamls_run folder on a single GPU
python main_run_folder.py --devices 0
# run all yamls from the yamls_run folder on a 4 GPUs
python main_run_folder.py --devices 0,1,2,3
# run all yamls from the /myfolder folder on a 4 GPUs
python main_run_folder.py --devices 0,1,2,3 --folder /myfolder
```


# Run with SLURM

`main_sbatch.py` provides an utility to start runs on a SLURM managed cluster. You need to provide the max time of 
your job (via `--time`), the number of nodes (via `--nodes`, defaults to 1) and the hyperparameters for the run 
(via `--hp`).

```
# start a run on a single node
python main_sbatch.py --time 24:00:00 --hp vislstm/yamls/vil/pretrain/lstm_6M16_e400_bialter_bilatavg_lr1e3_res224.yaml
# train on 4 nodes
python main_sbatch.py --time 24:00:00 --nodes 4 --hp vislstm/yamls/vil/pretrain/lstm_6M16_e400_bialter_bilatavg_lr1e3_res224.yaml
```


# Use-cases
We provide example yaml files for the following use-cases (hyperparameters may be sub-optimal):
- Fine-tune a ImageNet-1K pre-trained model from torchhub on a custom dataset ([fine-tune a pre-trained ViL-T on CIFAR-10](https://github.com/NX-AI/vision-lstm/blob/main/src/vislstm/yamls/vil/transfer/finetune_from_torchhub.yaml))