import logging
import os
from pathlib import Path

import kappaprofiler as kp
import torch
import yaml
from torch.distributed import broadcast_object_list
from wandb.util import generate_id

from ksuit.callbacks import CallbackBase
from ksuit.configs import CliArgs, StaticConfig, Hyperparams, WandbConfig
from ksuit.distributed import (
    barrier,
    get_rank,
    get_local_rank,
    get_world_size,
    is_managed,
    run_unmanaged,
    run_managed,
    log_distributed_config,
    is_rank0,
    is_distributed,
)
from ksuit.factory import MasterFactory
from ksuit.models import LinearModel
from ksuit.providers import PathProvider, DatasetConfigProvider, MetricPropertyProvider
from ksuit.utils.data_container import DataContainer
from ksuit.utils.logging_utils import add_global_handlers, log_from_all_ranks
from ksuit.utils.pytorch_cuda_timing import cuda_start_event, cuda_end_event
from ksuit.utils.seed import set_seed
from ksuit.utils.system_info import get_cli_command, log_system_info
from ksuit.utils.version_check import check_versions
from ksuit.utils.wandb_utils import finish_wandb, init_wandb


class Runner:
    def run(self):
        # parse cli_args immediately for fast cli_args validation
        cli_args = CliArgs.from_cli_args()
        static_config = StaticConfig.from_uri(uri=cli_args.static_config_uri)
        # initialize loggers for setup
        add_global_handlers(log_file_uri=None)

        if is_managed():
            run_managed(
                accelerator=cli_args.accelerator,
                devices=cli_args.devices,
                main=self.main,
            )
        else:
            run_unmanaged(
                accelerator=cli_args.accelerator,
                devices=cli_args.devices,
                main=self.main,
                master_port=cli_args.master_port or static_config.master_port,
                mig_devices=static_config.mig_config,
            )

    @staticmethod
    def main(device):
        cli_args = CliArgs.from_cli_args()
        static_config = StaticConfig.from_uri(uri=cli_args.static_config_uri)
        add_global_handlers(log_file_uri=None)
        with log_from_all_ranks():
            logging.info(f"initialized process rank={get_rank()} local_rank={get_local_rank()} pid={os.getpid()}")
        barrier()
        logging.info(f"initialized {get_world_size()} processes")

        # CUDA_LAUNCH_BLOCKING=1 for debugging
        # os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)

        # cudnn
        if cli_args.accelerator == "gpu":
            if cli_args.cudnn_benchmark or static_config.default_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                assert not static_config.default_cudnn_deterministic, "cudnn_benchmark can make things non-deterministic"
            else:
                logging.warning(f"disabled cudnn benchmark")
                if static_config.default_cudnn_deterministic:
                    torch.backends.cudnn.deterministic = True
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                    logging.warning(f"enabled cudnn deterministic")

        # profiling
        is_cuda_profiling = False
        if cli_args.accelerator == "gpu":
            if cli_args.cuda_profiling or static_config.default_cuda_profiling:
                is_cuda_profiling = True
                kp.setup_async(cuda_start_event, cuda_end_event)
                logging.info(f"initialized profiler to call sync cuda")
        else:
            kp.setup_async_as_sync()

        # load hyperparameters
        stage_hp = Hyperparams.get_stage_hp(
            cli_args.hp,
            testrun=cli_args.testrun,
            minmodelrun=cli_args.minmodelrun,
            mindatarun=cli_args.mindatarun,
            mindurationrun=cli_args.mindurationrun,
        )

        # set environment variables
        for key, value in stage_hp.get("env", {}).items():
            os.environ[key] = value if isinstance(value, str) else str(value)

        # set MasterFactory base_path
        if "master_factory_base_path" in stage_hp:
            MasterFactory.add_base_path(stage_hp["master_factory_base_path"])

        # resume
        if cli_args.resume_stage_id is not None:
            assert "initializer" not in stage_hp["trainer"]
            if cli_args.resume_checkpoint is None:
                checkpoint = "latest"
            elif cli_args.resume_checkpoint.startswith("E"):
                checkpoint = dict(epoch=int(cli_args.resume_checkpoint[1:]))
            elif cli_args.resume_checkpoint.startswith("U"):
                checkpoint = dict(update=int(cli_args.resume_checkpoint[1:]))
            elif cli_args.resume_checkpoint.startswith("S"):
                checkpoint = dict(sample=int(cli_args.resume_checkpoint[1:]))
            else:
                # any checkpoint (like cp=last or cp=best.accuracy1.test.main)
                checkpoint = cli_args.resume_checkpoint
            stage_hp["trainer"]["initializer"] = dict(
                kind="resume_initializer",
                stage_id=cli_args.resume_stage_id,
                checkpoint=checkpoint,
            )

        # retrieve stage_id from hp (allows queueing up dependent stages by hardcoding stage_ids in the yamls) e.g.:
        # - pretrain MAE with stageid abcdefgh
        # - finetune MAE where the encoder is initialized with the encoder from stage_id abcdefgh
        stage_id = stage_hp.get("stage_id", None)
        # generate stage_id and sync across devices
        if stage_id is None:
            stage_id = generate_id()
            if is_distributed():
                object_list = [stage_id] if is_rank0() else [None]
                broadcast_object_list(object_list)
                stage_id = object_list[0]
        stage_name = stage_hp.get("stage_name", "default_stage")

        # initialize logging
        path_provider = PathProvider(
            output_path=static_config.output_path,
            model_path=static_config.model_path,
            stage_name=stage_name,
            stage_id=stage_id,
        )
        message_counter = add_global_handlers(log_file_uri=path_provider.logfile_uri)

        # init seed
        run_name = cli_args.name or stage_hp.pop("name", None)
        seed = stage_hp.pop("seed", None)
        if seed is None:
            seed = 0
            logging.info(f"no seed specified -> using seed={seed}")

        # initialize wandb
        wandb_config_uri = stage_hp.pop("wandb", None)
        if wandb_config_uri == "disabled":
            wandb_mode = "disabled"
        else:
            wandb_mode = cli_args.wandb_mode or static_config.default_wandb_mode
        if wandb_mode == "disabled":
            wandb_config_dict = {}
            if cli_args.wandb_config is not None:
                logging.warning(f"wandb_config is defined via CLI but mode is disabled -> wandb_config is not used")
            if wandb_config_uri is not None:
                logging.warning(f"wandb_config is defined via yaml but mode is disabled -> wandb_config is not used")
        else:
            # retrieve wandb config from yaml
            if wandb_config_uri is not None:
                wandb_config_uri = Path("wandb_configs") / wandb_config_uri
                if cli_args.wandb_config is not None:
                    logging.warning(f"wandb_config is defined via CLI and via yaml -> wandb_config from yaml is used")
            # retrieve wandb config from --wandb_config cli arg
            elif cli_args.wandb_config is not None:
                wandb_config_uri = Path("wandb_configs") / cli_args.wandb_config
            # use default wandb_config file
            else:
                wandb_config_uri = Path("wandb_config.yaml")
            with open(wandb_config_uri.with_suffix(".yaml")) as f:
                wandb_config_dict = yaml.safe_load(f)
        wandb_config = WandbConfig(mode=wandb_mode, **wandb_config_dict)
        metric_property_provider = MetricPropertyProvider()
        config_provider, summary_provider = init_wandb(
            device=device,
            run_name=run_name,
            stage_hp=stage_hp,
            wandb_config=wandb_config,
            path_provider=path_provider,
            metric_property_provider=metric_property_provider,
            account_name=static_config.account_name,
            tags=stage_hp.pop("tags", None),
            notes=stage_hp.pop("notes", None),
            group=stage_hp.pop("group", None),
            group_tags=stage_hp.pop("group_tags", None),
        )
        # log codebase "high-level" version name (git commit is logged anyway)
        # outside git repo -> "fatal: not a git repository (or any of the parent directories): .git"
        # no error handling required as tag is simply "" if not within git repo
        config_provider["code/tag"] = os.popen("git describe --abbrev=0").read().strip()

        # log setup
        logging.info("------------------")
        logging.info(f"stage_id: {stage_id}")
        logging.info(get_cli_command())
        check_versions(verbose=True)
        log_system_info()
        static_config.log()
        cli_args.log()
        log_distributed_config()
        Hyperparams.log_stage_hp(stage_hp)
        if is_rank0():
            Hyperparams.save_unresolved_hp(cli_args.hp, path_provider.stage_output_path / "hp_unresolved.yaml")
            Hyperparams.save_resolved_hp(stage_hp, path_provider.stage_output_path / "hp_resolved.yaml")

        logging.info("------------------")
        logging.info(f"training stage '{path_provider.stage_name}'")
        if is_distributed():
            # using a different seed for every rank to ensure that stochastic processes are different across ranks
            # for large batch_sizes this shouldn't matter too much
            # this is relevant for:
            # - augmentations (augmentation parameters of sample0 of rank0 == augparams of sample0 of rank1 == ...)
            # - the masks of a MAE are the same for every rank
            # NOTE: DDP syncs the parameters in its __init__ method -> same initial parameters independent of seed
            seed += get_rank()
            logging.info(f"using different seeds per process (seed+rank)")
        set_seed(seed)

        # init datasets
        logging.info("------------------")
        logging.info("initializing datasets")
        datasets = {}
        dataset_config_provider = DatasetConfigProvider(
            global_dataset_paths=static_config.global_dataset_paths,
            local_dataset_path=static_config.local_dataset_path,
            data_source_modes=static_config.data_source_modes,
        )
        for dataset_key, dataset_kwargs in stage_hp["datasets"].items():
            logging.info(f"initializing {dataset_key}")
            datasets[dataset_key] = MasterFactory.get("dataset").instantiate(
                **dataset_kwargs,
                optional_kwargs=dict(
                    dataset_config_provider=dataset_config_provider,
                    path_provider=path_provider,
                ),
            )
        data_container_kwargs = {}
        if "prefetch_factor" in stage_hp:
            data_container_kwargs["prefetch_factor"] = stage_hp.pop("prefetch_factor")
        if "max_num_workers" in stage_hp:
            data_container_kwargs["max_num_workers"] = stage_hp.pop("max_num_workers")
        data_container = DataContainer(
            **datasets,
            num_workers=cli_args.num_workers,
            pin_memory=cli_args.pin_memory,
            config_provider=config_provider,
            **data_container_kwargs,
        )

        # init trainer
        logging.info("------------------")
        logging.info("initializing trainer")
        trainer = MasterFactory.get("trainer").instantiate(
            **stage_hp["trainer"],
            optional_kwargs=dict(
                data_container=data_container,
                device=device,
                sync_batchnorm=cli_args.sync_batchnorm or static_config.default_sync_batchnorm,
                config_provider=config_provider,
                summary_provider=summary_provider,
                path_provider=path_provider,
            ),
        )
        # register datasets of callbacks (e.g. for ImageNet-C the dataset never changes so its pointless to specify)
        for callback in trainer.callbacks:
            callback.register_root_datasets(
                dataset_config_provider=dataset_config_provider,
                is_mindatarun=cli_args.testrun or cli_args.mindatarun,
            )

        # init model
        logging.info("------------------")
        logging.info("creating model")
        if "model" not in stage_hp:
            logging.info(f"no model defined -> use linear model")
            model = LinearModel(
                input_shape=trainer.input_shape,
                output_shape=trainer.output_shape,
                update_counter=trainer.update_counter,
                path_provider=path_provider,
                is_frozen=True,
            )
        else:
            model = MasterFactory.get("model").instantiate(
                **stage_hp["model"],
                optional_kwargs=dict(
                    input_shape=trainer.input_shape,
                    output_shape=trainer.output_shape,
                    update_counter=trainer.update_counter,
                    path_provider=path_provider,
                    data_container=data_container,
                ),
            )
        logging.info(f"model:\n{model}")

        # train model
        trainer.train(model)

        # finish callbacks
        CallbackBase.finish()

        # summarize logvalues
        logging.info("------------------")
        logging.info(f"summarize logvalues")
        summary_provider.summarize_logvalues()
        summary_provider.flush()

        # log profiler times
        if not is_cuda_profiling:
            logging.warning(
                f"cuda profiling is not activated -> all cuda calls are executed asynchronously -> "
                f"this will result in inaccurate profiling times where the time for all asynchronous cuda operation "
                f"will be attributed to the first synchronous cuda operation "
                f"https://github.com/BenediktAlkin/KappaProfiler?tab=readme-ov-file#time-async-operations"
            )
        logging.info(f"full profiling times:\n{kp.profiler.to_string()}")
        kp.reset()

        # cleanup
        logging.info("------------------")
        logging.info(f"CLEANUP")
        message_counter.log()
        finish_wandb(wandb_config)
