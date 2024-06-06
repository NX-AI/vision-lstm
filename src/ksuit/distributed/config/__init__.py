from .base_config import BaseConfig
from .default_config import DefaultConfig

_config: BaseConfig = DefaultConfig()


def set_config(new_config: BaseConfig):
    global _config
    _config = new_config


def is_managed():
    return _config.is_managed()


def get_local_rank():
    return _config.get_local_rank()


def get_num_nodes():
    return _config.get_num_nodes()


def get_managed_world_size():
    return _config.get_managed_world_size()


def get_managed_rank():
    return _config.get_managed_rank()


def is_distributed():
    return _config.is_distributed()


def get_rank():
    return _config.get_rank()


def get_world_size():
    return _config.get_world_size()


def is_data_rank0():
    return _config.is_data_rank0()


def is_rank0():
    return _config.is_rank0()


def is_local_rank0():
    return _config.is_local_rank0()


def barrier():
    return _config.barrier()


def is_own_work(idx):
    return _config.is_own_work(idx=idx)


def get_backend():
    return _config.get_backend()


def log_distributed_config():
    return _config.log_distributed_config()
