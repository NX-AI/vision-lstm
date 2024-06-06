import logging
import os
import platform
from copy import deepcopy

import torch
import wandb

from ksuit.configs import WandbConfig
from ksuit.distributed import is_rank0, get_world_size, get_num_nodes, get_rank
from ksuit.providers import (
    NoopConfigProvider,
    PrimitiveConfigProvider,
    WandbConfigProvider,
    PathProvider,
    NoopSummaryProvider,
    PrimitiveSummaryProvider,
    WandbSummaryProvider,
    MetricPropertyProvider,
)


def init_wandb(
        device: str,
        run_name: str,
        stage_hp: dict,
        wandb_config: WandbConfig,
        path_provider: PathProvider,
        metric_property_provider: MetricPropertyProvider,
        account_name: str,
        tags: list,
        notes: str,
        group: str,
        group_tags: dict,
):
    logging.info("------------------")
    logging.info(f"initializing wandb (mode={wandb_config.mode})")
    # os.environ["WANDB_SILENT"] = "true"

    # create config_provider & summary_provider
    if not is_rank0():
        config_provider = NoopConfigProvider()
        summary_provider = NoopSummaryProvider()
        return config_provider, summary_provider
    elif wandb_config.is_disabled:
        config_provider = PrimitiveConfigProvider(path_provider=path_provider)
        summary_provider = PrimitiveSummaryProvider(
            path_provider=path_provider,
            metric_property_provider=metric_property_provider,
        )
    else:
        config_provider = WandbConfigProvider(path_provider=path_provider)
        summary_provider = WandbSummaryProvider(
            path_provider=path_provider,
            metric_property_provider=metric_property_provider,
        )

    config = dict(
        run_name=run_name,
        stage_name=path_provider.stage_name,
        hp=_lists_to_dict(stage_hp),
    )
    if not wandb_config.is_disabled:
        if wandb_config.mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        logging.info(f"logging into wandb (host={wandb_config.host} rank={get_rank()})")
        wandb.login(host=wandb_config.host)
        logging.info(f"logged into wandb (host={wandb_config.host})")
        name = run_name or "None"
        if path_provider.stage_name != "default_stage":
            name += f"/{path_provider.stage_name}"
        wandb_id = path_provider.stage_id
        # can't group by tags -> with group tags you can (by adding it as a field to the config)
        # group_tags:
        #   augmentation: minimal
        #   ablation: warmup
        tags = tags or []
        if group_tags is not None and len(group_tags) > 0:
            logging.info(f"group tags:")
            for group_name, tag in group_tags.items():
                logging.info(f"  {group_name}: {tag}")
                assert tag not in tags, \
                    f"tag '{tag}' from group_tags is also in tags (group_tags={group_tags} tags={tags})"
                tags.append(tag)
                config[group_name] = tag
        if len(tags) > 0:
            logging.info(f"tags:")
            for tag in tags:
                logging.info(f"- {tag}")
        wandb.init(
            entity=wandb_config.entity,
            project=wandb_config.project,
            name=name,
            dir=str(path_provider.stage_output_path),
            save_code=False,
            config=config,
            mode=wandb_config.mode,
            id=wandb_id,
            # add default tag to mark runs which have not been looked at in W&B
            # ints need to be cast to string
            tags=["new"] + [str(tag) for tag in tags],
            notes=notes,
            group=group or wandb_id,
        )
    config_provider.update(config)

    # log additional environment properties
    additional_config = {}
    if str(device) == "cpu":
        additional_config["device"] = "cpu"
    else:
        additional_config["device"] = torch.cuda.get_device_name(0)
    additional_config["dist/world_size"] = get_world_size()
    additional_config["dist/nodes"] = get_num_nodes()
    # hostname from static config which can be more descriptive than the platform.uname().node (e.g. account name)
    additional_config["dist/account_name"] = account_name
    additional_config["dist/hostname"] = platform.uname().node
    if "SLURM_JOB_ID" in os.environ:
        additional_config["dist/jobid"] = os.environ["SLURM_JOB_ID"]
    if "PBS_JOBID" in os.environ:
        additional_config["dist/jobid"] = os.environ["PBS_JOBID"]
    config_provider.update(additional_config)

    return config_provider, summary_provider


def _lists_to_dict(root):
    """ wandb cant handle lists in configs -> transform lists into dicts with str(i) as key """
    #  (it will be displayed as [{"kind": "..."}, ...])
    root = deepcopy(root)
    return _lists_to_dicts_impl(dict(root=root))["root"]


def _lists_to_dicts_impl(root):
    if not isinstance(root, dict):
        return
    for k, v in root.items():
        if isinstance(v, list):
            root[k] = {str(i): vitem for i, vitem in enumerate(v)}
        elif isinstance(v, dict):
            root[k] = _lists_to_dicts_impl(root[k])
    return root


def finish_wandb(wandb_config: WandbConfig):
    if not is_rank0() or wandb_config.is_disabled:
        return
    wandb.finish()
