import logging
import os
import random
from pathlib import Path
from typing import Optional

import kappaconfig as kc

from ksuit.distributed.config import is_distributed
from ksuit.utils.param_checking import to_path
from .wandb_config import WandbConfig


class StaticConfig:
    @staticmethod
    def from_uri(uri: str, template_path="./setup"):
        uri = to_path(uri)
        config = kc.DefaultResolver(template_path=template_path).resolve(kc.from_file_uri(uri))
        return StaticConfig(config)

    def __init__(self, config: dict):
        self.logger = logging.getLogger(type(self).__name__)
        self._config = config
        for key, value in config.get("env", {}).items():
            os.environ[key] = str(value)

    # region param checking
    def _check_bool(self, key, default):
        if key not in self._config:
            assert isinstance(default, bool)
            return default
        value = self._config[key]
        assert isinstance(value, bool), f"{key} {value} is not a bool"
        return value

    # endregion

    @property
    def account_name(self) -> str:
        if "account_name" not in self._config:
            return "anonymous"
        return self._config["account_name"]

    @property
    def output_path(self) -> Path:
        assert "output_path" in self._config, f"no 'output_path' defined in static_config"
        return to_path(self._config["output_path"])

    @property
    def model_path(self) -> Optional[Path]:
        return to_path(self._config.get("model_path", None))

    # region dataset
    @property
    def global_dataset_paths(self) -> dict:
        return self._config.get("global_dataset_paths", {})

    @property
    def local_dataset_path(self) -> Path:
        if "local_dataset_path" not in self._config:
            return None
        path = to_path(self._config["local_dataset_path"], check_exists=False)
        path.mkdir(exist_ok=True)
        return path

    @property
    def data_source_modes(self) -> dict:
        if "data_source_modes" not in self._config:
            return {}
        data_source_modes = self._config["data_source_modes"]
        assert all(data_source_mode in ["global", "local"] for data_source_mode in data_source_modes.values())
        return data_source_modes

    # endregion

    @property
    def mig_config(self) -> dict:
        if "mig" not in self._config:
            return {}
        mig = self._config["mig"]
        # mig is mapping from hostnames to devices to MIG-IDS
        # hostname:
        #   0: MIG-abcdef-ghi...
        assert isinstance(mig, dict), f"mig {mig} is not dict"
        for hostname, device_to_migid in mig.items():
            assert isinstance(hostname, str), f"hostnames should be strings (got {hostname})"
            assert isinstance(device_to_migid, dict), f"devices_to_migid should be dict (got {device_to_migid})"
            for device_idx, mig_id in device_to_migid.items():
                assert isinstance(device_idx, int), f"devices_to_migid keys should be int (got {device_idx})"
                assert isinstance(mig_id, str), f"devices_to_migid values should be str (got {mig_id})"
        return mig

    @property
    def default_wandb_mode(self) -> str:
        if "default_wandb_mode" not in self._config:
            return "disabled"
        mode = self._config["default_wandb_mode"]
        assert mode in WandbConfig.MODES, f"default_wandb_mode '{mode}' not in {WandbConfig.MODES}"
        return mode

    # region deterministic/profiling
    @property
    def default_cudnn_benchmark(self) -> bool:
        return self._check_bool("default_cudnn_benchmark", default=True)

    @property
    def default_cudnn_deterministic(self) -> bool:
        return self._check_bool("default_cudnn_deterministic", default=False)

    @property
    def default_cuda_profiling(self) -> bool:
        return self._check_bool("default_cuda_profiling", default=False)

    # endregion

    # region distributed
    @property
    def default_sync_batchnorm(self) -> bool:
        return self._check_bool("default_sync_batchnorm", default=True)

    @property
    def master_port(self):
        if "MASTER_PORT" in os.environ:
            return int(os.environ["MASTER_PORT"])
        if "master_port" not in self._config:
            return random.Random().randint(20000, 60000)
        master_port = self._config["master_port"]
        if not isinstance(master_port, int):
            master_port_min, master_port_max = master_port
            return random.Random().randint(master_port_min, master_port_max)
        assert isinstance(master_port, int), f"master_port is not an int ({master_port})"
        return master_port

    # endregion

    def log(self, verbose=False):
        self.logger.info("------------------")
        self.logger.info("STATIC CONFIG")
        self.logger.info(f"account_name: {self.account_name}")
        self.logger.info(f"output_path: {self.output_path}")
        # datasets
        if verbose:
            self.logger.info(f"global_dataset_paths:")
            for key, dataset_path in self._config["global_dataset_paths"].items():
                self.logger.info(f"  {key}: {Path(dataset_path).expanduser()}")
        if "local_dataset_path" in self._config:
            self.logger.info(f"local_dataset_path: {self._config['local_dataset_path']}")
            if os.name == "posix":
                # log available space on local disk
                self.logger.info(f"available space in local_dataset_path:")
                for line in os.popen(f"df -h {self._config['local_dataset_path']}").read().strip().split("\n"):
                    self.logger.info(line)
        if "data_source_modes" in self._config:
            self.logger.info(f"data_source_modes:")
            for key, source_mode in self._config["data_source_modes"].items():
                self.logger.info(f"  {key}: {source_mode}")
        # other
        if verbose:
            env = self._config.get("env", {})
            if len(env) > 0:
                self.logger.info("env:")
                for key, value in env.items():
                    self.logger.info(f"  {key}: {value}")
            self.logger.info(f"default_wandb_mode: {self.default_wandb_mode}")
            self.logger.info(f"default_cudnn_benchmark: {self.default_cudnn_benchmark}")
            self.logger.info(f"default_cudnn_deterministic: {self.default_cudnn_deterministic}")
            self.logger.info(f"default_cuda_profiling: {self.default_cuda_profiling}")
            # distributed
            if is_distributed():
                self.logger.info(f"master_port: {self.master_port}")
                self.logger.info(f"default_sync_batchnorm: {self.default_sync_batchnorm}")
