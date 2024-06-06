import logging
import os

from .base_config import BaseConfig


class DefaultConfig(BaseConfig):
    def is_managed(self) -> bool:
        if os.environ.get("SLURM_JOB_NAME", None) == "interactive":
            return False
        return "SLURM_PROCID" in os.environ

    def get_local_rank(self) -> int:
        if os.environ.get("SLURM_JOB_NAME", None) == "interactive":
            return self.get_rank()
        if "SLURM_LOCALID" in os.environ:
            return int(os.environ["SLURM_LOCALID"])
        return self.get_rank()

    def get_num_nodes(self) -> int:
        if "SLURM_JOB_NUM_NODES" in os.environ:
            return int(os.environ["SLURM_JOB_NUM_NODES"])
        return 1

    def get_managed_world_size(self) -> int:
        if "SLURM_NTASKS_PER_NODE" in os.environ:
            return self.get_num_nodes() * int(os.environ["SLURM_NTASKS_PER_NODE"])
        raise NotImplementedError

    def get_managed_rank(self) -> int:
        if "SLURM_PROCID" in os.environ:
            return int(os.environ["SLURM_PROCID"])
        raise NotImplementedError

    def log_distributed_config(self) -> None:
        logging.info("------------------")
        logging.info("DIST CONFIG")
        logging.info(f"rank: {self.get_rank()}")
        logging.info(f"local_rank: {self.get_local_rank()}")
        logging.info(f"world_size: {self.get_world_size()}")
        logging.info(f"nodes: {self.get_num_nodes()}")
        logging.info(f"backend: {self.get_backend()}")
        if "SLURM_JOB_ID" in os.environ:
            logging.info(f"slurm job id: {os.environ['SLURM_JOB_ID']}")
        if "ALL_HOST_NAMES" in os.environ:
            logging.info(f"hostnames: {os.environ['ALL_HOST_NAMES']}")
