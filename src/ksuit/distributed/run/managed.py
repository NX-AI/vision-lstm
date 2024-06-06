import logging
import os
import platform

from torch.distributed import init_process_group, destroy_process_group, barrier

from ksuit.distributed.config import (
    is_managed,
    get_managed_world_size,
    get_managed_rank,
    get_local_rank,
    get_num_nodes,
)
from .utils import check_single_device_visible, accelerator_to_device, get_backend


def run_managed(main, accelerator="gpu", devices=None):
    assert is_managed()
    # some HPCs dont set CUDA_VISIBLE_DEVICES at all (e.g. lux)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        local_rank = get_local_rank()
        visible_device_str = str(local_rank)
        logging.info(
            f"no CUDA_VISIBLE_DEVICES found "
            f"-> set CUDA_VISIBLE_DEVICES={visible_device_str} (local_rank={local_rank})"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_device_str
    else:
        # srun doesnt set correct CUDA_VISIBLE_DEVICES
        split = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(split) > 1:
            local_rank = get_local_rank()
            visible_device_str = split[local_rank]
            logging.info(
                f"found multiple visible devices (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}) "
                f"-> set CUDA_VISIBLE_DEVICES={visible_device_str} (local_rank={local_rank})"
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_device_str
    check_single_device_visible(accelerator=accelerator)
    assert devices is None, f"devices are set implicitly via environment"
    world_size = get_managed_world_size()
    if world_size == 1:
        # no need for setting up distributed stuff
        _run_managed_singleprocess(accelerator, main)
    else:
        # use all GPUs for training
        _run_managed_multiprocess(accelerator, main)


def _run_managed_singleprocess(accelerator, main):
    # single process
    logging.info(f"running single process slurm training")
    device = accelerator_to_device(accelerator)
    main(device=device)


def _run_managed_multiprocess(accelerator, main):
    # setup MASTER_ADDR & MASTER_PORT
    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ

    # get config from env variables
    world_size = get_managed_world_size()
    rank = get_managed_rank()

    # init process group
    logging.info(
        f"initializing rank={rank} local_rank={get_local_rank()} "
        f"nodes={get_num_nodes()} hostname={platform.uname().node} "
        f"master_addr={os.environ['MASTER_ADDR']} master_port={os.environ['MASTER_PORT']} "
        f"(waiting for all {world_size} processes to connect)"
    )
    init_process_group(backend=get_backend(accelerator), init_method="env://", world_size=world_size, rank=rank)
    barrier()

    # start main_single
    device = accelerator_to_device(accelerator)
    main(device=device)
    destroy_process_group()
