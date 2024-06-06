import logging
import os
import platform

import yaml


def check_single_device_visible(accelerator):
    if accelerator == "cpu":
        # nothing to check
        return
    elif accelerator == "gpu":
        # if "import torch" is called before "CUDA_VISIBLE_DEVICES" is set, torch will see all devices
        assert "CUDA_VISIBLE_DEVICES" in os.environ
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1, f"{os.environ['CUDA_VISIBLE_DEVICES']}"
        import torch
        assert torch.cuda.is_available(), f"CUDA not available use --accelerator cpu to run on cpu"
        visible_device_count = torch.cuda.device_count()
        assert visible_device_count <= 1, \
            f"set CUDA_VISIBLE_DEVICES before importing torch " \
            f"CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}' " \
            f"torch.cuda.device_count={visible_device_count}"
    else:
        raise NotImplementedError


def get_backend(accelerator, device_ids=None):
    if accelerator == "cpu":
        # gloo is recommended for cpu multiprocessing
        # https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
        return "gloo"
    if os.name == "nt":
        # windows doesn't support nccl
        return "gloo"
    # MIG doesn't support NCCL
    if device_ids is not None:
        for device_id in device_ids:
            try:
                int(device_id)
            except ValueError:
                return "gloo"
    # nccl is recommended for gpu multiprocessing
    # https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
    return "nccl"


def accelerator_to_device(accelerator):
    if accelerator == "cpu":
        return "cpu"
    elif accelerator == "gpu":
        return "cuda"
    raise NotImplementedError


def parse_devices(accelerator, devices, mig_devices=None):
    # if devices is not defined -> use all devices on node
    if accelerator == "gpu" and devices is None:
        logging.info("no device subset defined -> use all devices on node")
        # retrieve device names via nvidia-smi because CUDA_VISIBLE_DEVICES needs
        # to be set before calling anything in torch.cuda -> only 1 visible device
        all_devices = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read().strip().split("\n")
        device_ids = [str(i) for i in range(len(all_devices))]
        logging.info(f"found {len(device_ids)} device(s)")
        # not supported with MIG
        nvidia_smi_output = os.popen("nvidia-smi").read().strip().split("\n")
        assert all("MIG device" not in line for line in nvidia_smi_output), \
            "using all devices on node is not supported with MIG"
    else:
        try:
            # single process
            device_ids = [int(devices)]
        except ValueError:
            # multi process
            device_ids = yaml.safe_load(f"[{devices}]")
            assert all(isinstance(d, int) for d in device_ids), \
                f"invalid devices specification '{devices}' (specify multiple devices like this '0,1,2,3')"
        # os.environ["CUDA_VISIBLE_DEVICES"] requires string
        device_ids = [str(device_id) for device_id in device_ids]

        if accelerator == "gpu" and mig_devices is not None:
            # map to MIG device ids
            hostname = platform.uname().node
            if hostname in mig_devices:
                for i in range(len(device_ids)):
                    device_id = int(device_ids[i])
                    if device_id in mig_devices[hostname]:
                        mig_device_id = mig_devices[hostname][device_id]
                        device_ids[i] = mig_device_id
                        logging.info(f"device_id is MIG device with id {mig_device_id}")

    return len(device_ids), device_ids


def log_device_info(accelerator, device_ids):
    if accelerator == "cpu":
        for i in range(len(device_ids)):
            logging.info(f"device {i}: cpu")
    elif accelerator == "gpu":
        # retrieve device names via nvidia-smi because CUDA_VISIBLE_DEVICES needs to be set before calling anything
        # in torch.cuda -> only 1 visible device
        all_devices = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read().strip().split("\n")
        for i, device_id in enumerate(device_ids):
            try:
                device_id = int(device_id)
                logging.info(f"device {i}: {all_devices[device_id]} (id={device_id})")
            except ValueError:
                # MIG device
                logging.info(f"using MIG device")
    else:
        raise NotImplementedError
