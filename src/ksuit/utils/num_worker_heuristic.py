import logging
import os
from ksuit.distributed import is_managed

def get_fair_cpu_count():
    total_cpu_count = get_total_cpu_count()
    if total_cpu_count == 0:
        return 0

    device_count = _get_device_count()
    # use SLURM definitions if available
    if is_managed():
        # slurm already divides cpus among tasks
        tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        if "SLURM_CPUS_PER_TASK" in os.environ:
            cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
        elif "SLURM_CPUS_ON_NODE" in os.environ:
            cpus_on_node = int(os.environ["SLURM_CPUS_ON_NODE"])
            cpus_per_task = cpus_on_node // tasks_per_node
        else:
            # fall back to non-slurm distribution of CPUs
            cpus_per_task = None
            logging.warning(f"SLURM_NTASKS_PER_NODE not defined -> divide CPUs equally among GPUs")
        if cpus_per_task is not None:
            # currently only 1 GPU per task is supported
            assert device_count == tasks_per_node, f"{device_count} != {tasks_per_node}"
            if total_cpu_count != cpus_per_task:
                logging.warning(f"total_cpu_count != cpus_per_task ({total_cpu_count} != {cpus_per_task})")
            return cpus_per_task - 1
    # divide cpus among devices
    return int(total_cpu_count / device_count)


def _get_device_count():
    # get number of devices per node (srun nvidia-smi shows all devices not only the ones assigned for the srun task)
    # (if no GPU is available this returns "")
    # normal example output:
    # GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-...)
    # GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-...)
    # MIG example output:
    # GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-...)
    #   MIG 3g.20gb     Device  0: (UUID: MIG-...)
    #   MIG 3g.20gb     Device  1: (UUID: MIG-...)
    # GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-...)
    #   MIG 3g.20gb     Device  0: (UUID: MIG-...)
    #   MIG 3g.20gb     Device  1: (UUID: MIG-...)
    nvidia_smi_lines = os.popen("nvidia-smi -L").read().strip().split("\n")

    # create dict from GPU to MIG devices:
    # {
    #   GPU0: 1 # normal GPU
    #   GPU1: 2 # split into 2 MIG devices
    # }
    devices_per_gpu = {}
    devices_counter = 0
    for i, line in enumerate(nvidia_smi_lines):
        if "MIG" in line:
            devices_counter += 1
        if "GPU" in line and i == 0 and len(nvidia_smi_lines) > 1 and "MIG" in nvidia_smi_lines[i + 1]:
            continue
        if "GPU" in line or i == len(nvidia_smi_lines) - 1:
            if devices_counter == 0:
                devices_counter = 1  # normal GPU -> single device
            devices_per_gpu[len(devices_per_gpu)] = devices_counter
            devices_counter = 0
    # count devices
    devices_on_node = sum(devices_per_gpu.values())

    if devices_on_node == 0:
        devices_on_node = 1
    return devices_on_node


def get_total_cpu_count():
    if os.name == "nt":
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        if cpu_count <= 16:
            # don't bother on dev machines
            return 0
    else:
        cpu_count = len(os.sched_getaffinity(0))

    return cpu_count
