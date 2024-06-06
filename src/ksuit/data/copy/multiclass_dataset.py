import os
import shutil
import zipfile
from pathlib import Path

from ksuit.utils.logging_utils import log
from .utils import folder_contains_mostly_zips, run_unzip_jobs, generic_global_to_local


def _check_global_root_exists(global_root):
    if global_root.exists() and global_root.is_dir():
        return True
    if global_root.with_suffix(".zip").exists():
        return True
    return False


def _copy_global_to_local(global_root, local_root, num_workers, log_fn):
    if global_root.exists() and global_root.is_dir():
        contains_mostly_zips, zips = folder_contains_mostly_zips(global_root)
        if contains_mostly_zips:
            # extract all zip folders into dst (e.g. imagenet1k/train/n01558993.zip)
            log(log_fn, f"extracting {len(zips)} zips from '{global_root}' to '{local_root}' ({num_workers} workers)")
            # compose jobs
            jobargs = []
            for item in os.listdir(global_root):
                if not item.endswith(".zip"):
                    continue
                dst_uri = (local_root / item).with_suffix("")
                src_uri = global_root / item
                jobargs.append((src_uri, dst_uri))
            # run jobs
            run_unzip_jobs(jobargs=jobargs, num_workers=num_workers)
        else:
            # copy folders which contain the raw files (not zipped or anything)
            log(log_fn, f"copying folders of '{global_root}' to '{local_root}'")
            # copy folder (dirs_exist_ok=True because local_root is created for start_copy_file)
            shutil.copytree(global_root, local_root, dirs_exist_ok=True)
    elif global_root.with_suffix(".zip").exists():
        log(log_fn, f"extracting '{global_root.with_suffix('.zip')}' to '{local_root}'")
        # extract zip
        with zipfile.ZipFile(global_root.with_suffix(".zip")) as f:
            f.extractall(local_root)
    else:
        raise NotImplementedError


def multiclass_global_to_local(
        global_root,
        local_root,
        num_workers=0,
        log_fn=None,
):
    generic_global_to_local(
        global_root=global_root,
        local_root=local_root,
        copy_global_to_local_fn=_copy_global_to_local,
        copy_global_to_local_kwargs=dict(num_workers=num_workers),
        check_global_root_exists_fn=_check_global_root_exists,
        log_fn=log_fn,
    )


def create_zips_splitwise(src, dst):
    """
    Creates a zip for each split.
    Source: imagenet1k/train/...
    Result: imagenet1k/train.zip
    :param src: Path to source folder (e.g. /data/imagenet1k/train)
    :param dst: Path to output folder (e.g. /data/imagenet1k_splitzip)
    """
    src = Path(src).expanduser()
    assert src.exists(), f"src '{src}' doesn't exist"
    dst = Path(dst).expanduser()
    assert not dst.exists()
    dst.mkdir(parents=True)
    shutil.make_archive(
        base_name=dst / src.name,
        format="zip",
        root_dir=src,
    )


def create_zips_classwise(src, dst):
    """
    creates a zip for each class
    Source:
    imagenet1k/train/n2933412
    imagenet1k/train/n3498534
    Result:
    imagenet1k/train/n2933412.zip
    imagenet1k/train/n3498534.zip
    """
    src = Path(src).expanduser()
    assert src.exists(), f"src '{src}' doesn't exist"
    dst = Path(dst).expanduser()
    assert not dst.exists()
    dst.mkdir(parents=True)

    for item in os.listdir(src):
        src_item = src / item
        if not src_item.is_dir():
            continue
        shutil.make_archive(
            base_name=dst / item,
            format="zip",
            root_dir=src_item,
        )
