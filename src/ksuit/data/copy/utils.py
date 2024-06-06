import os
import shutil
import zipfile
from time import sleep

import joblib
import psutil

from ksuit.utils.logging_utils import log
from ksuit.utils.param_checking import to_path


def generic_global_to_local(
        global_root,
        local_root,
        copy_global_to_local_fn,
        copy_global_to_local_kwargs=None,
        check_global_root_exists_fn=None,
        log_fn=None,
):
    global_root = to_path(global_root)
    local_root = to_path(local_root)

    if check_global_root_exists_fn is not None:
        check_global_root_exists_fn(global_root)

    # if local_path exists:
    # - autocopy start/end file exists -> already copied -> do nothing
    # - autocopy start file exists && autocopy end file doesn't exist -> incomplete copy -> delete and copy again
    # - autocopy start file doesn't exists -> manually copied dataset -> do nothing
    start_copy_file = local_root / "autocopy_start.txt"
    end_copy_file = local_root / "autocopy_end.txt"
    if local_root.exists():
        if start_copy_file.exists():
            if end_copy_file.exists():
                # already automatically copied -> do nothing
                log(log_fn, f"dataset was already automatically copied '{local_root}'")
                return
            else:
                # incomplete copy -> delete and copy again
                log(log_fn, f"found incomplete automatic copy in '{local_root}'")
                # check processid
                with open(start_copy_file) as f:
                    content = f.read()
                pid = int(content[content.index("pid=") + len("pid="):])
                if psutil.pid_exists(pid):
                    # give the process 10min to finish copying and check every minute
                    log(log_fn, f"found copying process is still working (pid={pid}) -> wait for a minute")
                    for i in range(10):
                        sleep(60)
                        if end_copy_file.exists():
                            log(log_fn, f"other process finished copying (pid={pid})")
                            return
                        log(log_fn, f"other process still copying (pid={pid}) -> wait for 60s ({i + 1}/10)")
                    raise RuntimeError("something went wrong during autocopying dataset")
                else:
                    log(log_fn, f"no running process to attempt to finish copying -> delete incomplete copy")
                    shutil.rmtree(local_root)
        else:
            log(log_fn, f"using manually copied dataset '{local_root}'")
            return
    local_root.mkdir(parents=True)

    # create start_copy_file
    with open(start_copy_file, "w") as f:
        f.write(f"an attempt to copy the dataset automatically was started from pid={os.getpid()}")

    # copy
    copy_global_to_local_fn(
        global_root=global_root,
        local_root=local_root,
        log_fn=log_fn,
        **(copy_global_to_local_kwargs or {}),
    )

    # create end_copy_file
    with open(end_copy_file, "w") as f:
        f.write("this file indicates that copying the dataset automatically was successful")

    log(log_fn, "finished copying data from global to local")


def folder_contains_mostly_zips(path):
    # check if subfolders are zips (allow files such as a README inside the folder)
    items = os.listdir(path)
    zips = [item for item in items if item.endswith(".zip")]
    contains_mostly_zips = len(zips) > 0 and len(zips) >= len(items) // 2
    return contains_mostly_zips, zips


def unzip(src, dst):
    with zipfile.ZipFile(src) as f:
        f.extractall(dst)


def run_unzip_jobs(jobargs, num_workers):
    if num_workers <= 1:
        for src, dst in jobargs:
            unzip(src, dst)
    else:
        jobs = [joblib.delayed(unzip)(src, dst) for src, dst in jobargs]
        pool = joblib.Parallel(n_jobs=num_workers)
        pool(jobs)
