import argparse
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path
from time import sleep

import yaml

from ksuit.utils.logging_utils import add_stdout_handler
from ksuit.utils.version_check import check_versions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="yamls_run")
    parser.add_argument("--devices", type=str, required=True)
    parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu"])
    parser.add_argument("--wandb_config", type=str)
    parser.add_argument("--wandb_mode", type=str)
    single_or_forever = parser.add_mutually_exclusive_group()
    single_or_forever.add_argument("--forever", action="store_true")
    single_or_forever.add_argument("--single", action="store_true")
    parser.add_argument("--start_on_idle", action="store_true")
    parser.add_argument("--num_workers", type=int)
    return vars(parser.parse_args())


def devices_are_idle(device_ids):
    used_vram = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader").read().strip().split("\n")
    for device_id in device_ids:
        split = used_vram[device_id].split(" ")
        # i think vram is always in MiB, but just to make sure
        assert split[1] == "MiB"
        value = int(split[0])
        if value > 10:
            logging.info(f"device {device_id} is in use")
            return False
    logging.info(f"devices {device_ids} are idle")
    return True


def main(
        folder,
        devices,
        accelerator,
        wandb_config,
        wandb_mode,
        forever,
        single,
        start_on_idle,
        num_workers,
):
    add_stdout_handler()

    # start when the devices are idle for at least 15 minutes
    if start_on_idle:
        device_ids = yaml.safe_load(f"[{devices}]")
        while True:
            if devices_are_idle(device_ids):
                # check frequently for the next 30 minutes if the device is still idle
                for _ in range(30):
                    sleep(60)
                    if not devices_are_idle(device_ids):
                        # devices were not idle for a longer time (probably started a new run)
                        break
                else:
                    # devices were idle for a longer time
                    break
            else:
                # wait 30 minutes for next check
                sleep(30 * 60)

    folder = Path(folder).expanduser()
    assert folder.exists()
    running_folder = folder / "running"
    finished_folder = folder / "finished"
    running_folder.mkdir(exist_ok=True)
    finished_folder.mkdir(exist_ok=True)
    counter = 0

    while True:
        # fetch list list of yamls to run
        folder_content = [folder / name for name in os.listdir(folder)]
        yaml_files = [entry for entry in folder_content if entry.is_file() and entry.name.endswith(".yaml")]
        if len(yaml_files) == 0:
            if forever:
                logging.info(f"no yamls in {folder} -> wait a minute")
                sleep(60)
                continue
            else:
                logging.info(f"no yamls in {folder} -> terminate")
                break

        # sleep for a random interval to avoid race conditions
        sleep(random.random())

        # pick random yaml and move it to running folder
        yaml_file = random.choice(yaml_files)
        if not yaml_file.exists():
            continue
        running_yaml = running_folder / yaml_file.name
        shutil.move(yaml_file, running_yaml)
        logging.info(f"moved {yaml_file} to {running_yaml}")

        # extract name from yaml (also implicitly checks if yaml is valid)
        # noinspection PyBroadException
        try:
            with open(running_yaml) as f:
                hp = yaml.safe_load(f)
            name = hp.get("name", None)
        except:
            logging.info(f"couldnt load yaml {yaml_file}")
            continue

        # start
        popen_arg_list = [
            "python", "main_train.py",
            "--hp", str(running_yaml),
            "--devices", devices,

        ]
        if name is None:
            popen_arg_list += ["--name", running_yaml.name]
        if wandb_config is not None:
            popen_arg_list += ["--wandb_config", wandb_config]
        if wandb_mode is not None:
            popen_arg_list += ["--wandb_mode", wandb_mode]
        if num_workers is not None:
            popen_arg_list += ["--num_workers", str(num_workers)]
        if accelerator is not None:
            popen_arg_list += ["--accelerator", accelerator]
        process = subprocess.Popen(popen_arg_list)
        logging.info(f"started {running_yaml.name}")
        process.wait()

        # move to finished folder
        shutil.move(running_yaml, finished_folder / yaml_file.name)
        counter += 1

        if single:
            break

    logging.info(f"finished running {counter} yamls from {folder} (devices={devices})")


if __name__ == "__main__":
    check_versions(verbose=False)
    main(**parse_args())
