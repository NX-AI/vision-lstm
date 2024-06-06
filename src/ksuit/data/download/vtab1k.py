import os
import shutil
import zipfile

import gdown

from ksuit.utils.param_checking import to_path


# TODO test
def download_vtab1k(root):
    root = to_path(root)
    if root.parent.exists():
        print(f"dataset was already downloaded -> skip ('{root.as_posix()}')")
        return
    print(
        f"'{root.as_posix()}' doesn't exist "
        f"-> download self-contained VTAB-1K dataset to {root.parent.as_posix()}"
    )
    root.parent.mkdir(exist_ok=True, parents=True)
    gdown.download(id="1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p", output=(root.parent / "vtab-1k.zip").as_posix())
    print(f"extracting downloaded dataset")
    with zipfile.ZipFile(root.parent / "vtab-1k.zip", "r") as f:
        f.extractall(root.parent)
    # zip contains a single folder -> would create .../vtab-1k/vtab-1k/cifar -> remove the folder
    for fname in os.listdir(root.parent / "vtab-1k"):
        shutil.move(root.parent / "vtab-1k" / fname, root.parent / fname)
    (root.parent / "vtab-1k").rmdir()
    os.remove(root.parent / "vtab-1k.zip")
