import logging
import os
import platform

from .utils import check_single_device_visible, log_device_info, accelerator_to_device, parse_devices, get_backend


def run_unmanaged(main, devices, accelerator="gpu", master_port=None, mig_devices=None):
    logging.info("------------------")
    # single node run
    world_size, device_ids = parse_devices(
        accelerator=accelerator,
        devices=devices,
        mig_devices=mig_devices,
    )
    if world_size == 1:
        # single process
        logging.info(f"running single process training")
        if accelerator == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[0]
        check_single_device_visible(accelerator=accelerator)
        log_device_info(accelerator=accelerator, device_ids=device_ids)
        device = accelerator_to_device(accelerator=accelerator)
        main(device=device)
    else:
        assert master_port is not None
        # spawn multi process training
        logging.info(
            f"running multi process training on {world_size} processes "
            f"(devices={devices} host={platform.uname().node})"
        )
        logging.info(f"master port: {master_port}")
        # dont log device info as this would load torch on device0 and block the VRAM required for this
        # log_device_info(accelerator, device_ids)
        args = (accelerator, device_ids, master_port, world_size, main)
        from torch.multiprocessing import spawn
        spawn(_run_multiprocess, nprocs=world_size, args=args)


def _run_multiprocess(rank, accelerator, device_ids, master_port, world_size, main):
    # unmanaged is limited to single-node -> use "localhost"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    if accelerator == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[rank]
    check_single_device_visible(accelerator=accelerator)

    from torch.distributed import init_process_group, destroy_process_group
    init_process_group(
        backend=get_backend(accelerator, device_ids),
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    device = accelerator_to_device(accelerator)
    main(device=device)
    destroy_process_group()
