import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from .wandb_config import WandbConfig


@dataclass
class CliArgs:
    hp: str
    accelerator: str
    devices: str
    num_workers: int
    pin_memory: bool
    wandb_mode: str
    wandb_config: str
    cudnn_benchmark: bool
    cuda_profiling: bool
    testrun: bool
    minmodelrun: bool
    mindatarun: bool
    mindurationrun: bool
    name: str
    static_config_uri: str
    master_port: int
    sync_batchnorm: bool
    resume_stage_id: str
    resume_checkpoint: str

    @staticmethod
    def from_cli_args():
        parser = ArgumentParser()
        parser.add_argument("--hp", type=_hp, required=True)
        parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu"])
        parser.add_argument("--devices", type=_devices)
        parser.add_argument("--name", type=str)
        parser.add_argument("--static_config_uri", type=str, default="static_config.yaml")
        # dataloading
        parser.add_argument("--num_workers", type=int)
        pin_memory_group = parser.add_mutually_exclusive_group()
        pin_memory_group.add_argument("--pin_memory", action="store_true")
        pin_memory_group.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
        pin_memory_group.set_defaults(pin_memory=None)
        # wandb
        parser.add_argument("--wandb_mode", type=str, choices=WandbConfig.MODES)
        parser.add_argument("--wandb_config", type=_wandb_config)
        # cudnn benchmark
        cudnn_benchmark_group = parser.add_mutually_exclusive_group()
        cudnn_benchmark_group.add_argument("--cudnn_benchmark", action="store_true")
        cudnn_benchmark_group.add_argument("--no_cudnn_benchmark", action="store_false", dest="cudnn_benchmark")
        cudnn_benchmark_group.set_defaults(cudnn_benchmark=None)
        # cuda profiling
        cuda_profiling_group = parser.add_mutually_exclusive_group()
        cuda_profiling_group.add_argument("--cuda_profiling", action="store_true")
        cuda_profiling_group.add_argument("--no_cuda_profiling", action="store_false", dest="cuda_profiling")
        cuda_profiling_group.set_defaults(cuda_profiling=None)
        # testrun
        testrun_group = parser.add_mutually_exclusive_group()
        testrun_group.add_argument("--testrun", action="store_true")
        testrun_group.add_argument("--minmodelrun", action="store_true")
        testrun_group.add_argument("--mindatarun", action="store_true")
        testrun_group.add_argument("--mindurationrun", action="store_true")
        # distributed
        parser.add_argument("--master_port", type=int)
        # distributed - syncbatchnorm
        sync_batchnorm_group = parser.add_mutually_exclusive_group()
        sync_batchnorm_group.add_argument("--sync_batchnorm", action="store_true")
        sync_batchnorm_group.add_argument("--local_batchnorm", action="store_false", dest="sync_batchnorm")
        sync_batchnorm_group.set_defaults(sync_batchnorm=None)
        # resume
        parser.add_argument("--resume_stage_id", type=str)
        parser.add_argument("--resume_checkpoint", type=str)

        return CliArgs(**vars(parser.parse_known_args()[0]))

    def log(self):
        logging.info("------------------")
        logging.info(f"CLI ARGS")
        for key, value in vars(self).items():
            if value is not None:
                logging.info(f"{key}: {value}")


def _hp(hp):
    assert isinstance(hp, str)
    path = Path(hp).expanduser().with_suffix(".yaml")
    assert path.exists(), f"hp file '{hp}' doesn't exist"
    return hp


def _devices(devices):
    assert isinstance(devices, str)
    if not devices.isdigit():
        assert all(d.isdigit() for d in devices.split(",")), f"specify multiple devices as 0,1,2,3 (not {devices})"
    return devices


def _wandb_config(wandb_config):
    if wandb_config is not None:
        assert isinstance(wandb_config, str)
        path = (Path("wandb_configs").expanduser() / wandb_config).with_suffix(".yaml")
        assert path.exists(), f"wandb_config file '{path}' doesn't exist"
        return wandb_config
