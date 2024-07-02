import os
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from ksuit.data import Dataset
from ksuit.providers import DatasetConfigProvider
from ksuit.distributed import is_data_rank0, barrier
from ksuit.utils.param_checking import to_path


class Ade20k(Dataset):
    @staticmethod
    def to_zip(src, dst=None):
        # Ade20k.to_zip(src="~/Documents/data/ade20k") will create "~/Documents/data/ade20k.zip"
        # Ade20k.to_zip(src="~/Documents/data/ade20k", dst="/Documents/ade20k.zip) will create "~/Documents/ade20k.zip"
        src = to_path(src)
        if dst is None:
            dst = src
        else:
            # require .zip for parameter but remove it because shutil.make_archive automatically appends .zip
            dst = to_path(dst, should_exist=False, suffix=".zip").with_suffix("")

        assert (src / "annotations").exists()
        assert (src / "images").exists()
        shutil.make_archive(
            base_name=dst,
            format="zip",
            root_dir=src,
        )

    @staticmethod
    def from_zip(src, dst):
        # Ade20k.from_zip(src="~/Documents/data/ade20k.zip", dst="~/Documents/data/ade20k")
        dst = to_path(dst, should_exist=False)
        src = to_path(src)
        assert src.exists() and src.as_posix().endswith(".zip"), f"invalid source zip: {src.as_posix()}"
        shutil.unpack_archive(
            filename=src,
            extract_dir=dst,
            format="zip",
        )
        Ade20k.check_valid(dst)

    @staticmethod
    def check_valid(root):
        root = to_path(root)
        assert (root / "annotations").exists()
        assert (root / "images").exists()

    def __init__(
            self,
            split,
            global_root=None,
            local_root=None,
            dataset_config_provider: DatasetConfigProvider = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if dataset_config_provider is not None:
            global_root, local_root = dataset_config_provider.get_roots(
                global_root=global_root,
                local_root=local_root,
                identifier="ade20k",
            )
        if local_root is None:
            # load from global root
            source_root = global_root
            assert global_root.exists(), f"invalid global_root '{global_root}'"
            self.logger.info(f"data_source (global): '{global_root}'")
        else:
            # load from local root
            source_root = local_root / "ade20k"
            if is_data_rank0():
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                if source_root.exists():
                    Ade20k.check_valid(source_root)
                else:
                    # copy data from global to local
                    global_root_contents = os.listdir(global_root)
                    # if zipfile exists -> extract zipfile
                    zip_src = global_root.with_suffix(".zip")
                    if zip_src.exists():
                        # extract zip
                        self.logger.info(f"extract {zip_src.as_posix()} to {source_root.as_posix()}")
                        Ade20k.from_zip(src=global_root.with_suffix(".zip"), dst=source_root)
                    elif "annotations" in global_root_contents and "images" in global_root_contents:
                        # copy annotations/images folder
                        self.logger.info(f"copy {global_root.as_posix()} to {source_root.as_posix()}")
                        source_root.mkdir()
                        shutil.copytree(global_root / "images", source_root / "images")
                        shutil.copytree(global_root / "annotations", source_root / "annotations")
                    else:
                        raise NotImplementedError
            barrier()

        self.split = split
        self.img_root = source_root / "images" / split
        self.mask_root = source_root / "annotations" / split
        self.fnames = list(sorted(os.listdir(self.img_root)))
        # images end with .jpg but annotations with .png
        assert all(fname.endswith(".jpg") for fname in self.fnames)
        self.fnames = [fname[:-len(".jpg")] for fname in self.fnames]

    def getitem_x(self, idx):
        return default_loader(self.img_root / f"{self.fnames[idx]}.jpg")

    def getitem_segmentation(self, idx):
        # 0 is background class and is typically ignored -> subtract 1 from all classes and ignore_index in loss
        return torch.from_numpy(np.array(Image.open(self.mask_root / f"{self.fnames[idx]}.png")).astype("int64")) - 1

    @staticmethod
    def getshape_segmentation():
        return 150,

    def __len__(self):
        return len(self.fnames)
