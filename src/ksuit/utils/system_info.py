import logging
import os
import platform
import shlex
import sys
from pathlib import Path

from ksuit.distributed import get_rank, get_local_rank
from ksuit.utils.logging_utils import log_from_all_ranks
from ksuit.utils.num_worker_heuristic import get_total_cpu_count


def get_cli_command():
    # print the command with which the script was called
    # https://stackoverflow.com/questions/37658154/get-command-line-arguments-as-string
    script_str = f"python {Path(sys.argv[0]).name}"
    argstr = " ".join(map(shlex.quote, sys.argv[1:]))
    return f"{script_str} {argstr}"


def get_installed_cuda_version():
    nvidia_smi_lines = os.popen("nvidia-smi").read().strip().split("\n")
    for line in nvidia_smi_lines:
        if "CUDA Version:" in line:
            return line[line.index("CUDA Version: ") + len("CUDA Version: "):-1].strip()
    return None


def log_system_info():
    logging.info("------------------")
    logging.info("SYSTEM INFO")
    logging.info(f"host name: {platform.uname().node}")
    logging.info(f"OS: {platform.platform()}")
    logging.info(f"OS version: {platform.version()}")
    cuda_version = get_installed_cuda_version()
    if cuda_version is not None:
        logging.info(f"CUDA version: {cuda_version}")

    # print hash of latest git commit (git describe or similar stuff is a bit ugly because it would require the
    # git.exe path to be added in path as conda/python do something with the path and don't use the system
    # PATH variable by default)
    git_hash_file = Path(".git") / "FETCH_HEAD"
    if git_hash_file.exists():
        with open(git_hash_file) as f:
            lines = f.readlines()
            if len(lines) == 0:
                # this happened when I didn't have internet
                logging.warning(f".git/FETCH_HEAD has no content")
            else:
                git_hash = lines[0][:40]
                logging.info(f"current commit hash: {git_hash}")
        git_tag = os.popen("git describe --abbrev=0").read().strip()
        logging.info(f"latest git tag: {git_tag}")
    else:
        logging.warning("could not retrieve current git commit hash from ./.git/FETCH_HEAD")
    with log_from_all_ranks():
        logging.info(
            f"initialized process rank={get_rank()} local_rank={get_local_rank()} pid={os.getpid()} "
            f"hostname={platform.uname().node}"
        )
    logging.info(f"total_cpu_count: {get_total_cpu_count()}")
