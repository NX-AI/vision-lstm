def torch_loader(uri):
    import torch
    return torch.load(uri)


def image_loader(uri):
    from torchvision.datasets.folder import default_loader
    return default_loader(uri)


SUFFIX_TO_LOADER = {
    "th": torch_loader,
    "pth": torch_loader,
    "torch": torch_loader,
    "pytorch": torch_loader,
    "png": image_loader,
    "jpg": image_loader,
    "jpeg": image_loader,
    "gif": image_loader,
}


def fname_to_loader(fname: str):
    suffix = fname.split(".")[-1].lower()
    assert suffix in SUFFIX_TO_LOADER, f"no loader defined for '{suffix}' files"
    return SUFFIX_TO_LOADER[suffix]
