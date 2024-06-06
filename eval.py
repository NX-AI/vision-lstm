import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode, ToTensor
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    return vars(parser.parse_args())


class NoopContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def main(data, name, device, batch_size, num_workers, precision):
    data = Path(data).expanduser()
    assert data.exists() and data.is_dir(), f"invalid data path '{data.as_posix()}'"

    # init device
    print(f"using device: {device}")
    device = torch.device(device)

    # init data
    print(f"initializing ImageNet-1K validation set '{data.as_posix()}'")
    dataset = ImageFolder(
        root=data,
        transform=Compose(
            [
                Resize(size=224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=224),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        ),
    )
    if os.name != "nt":
        assert len(dataset) == 50000, f"dataset is not ImageNet-1K validation set (len(dataset) = {len(dataset)})"

    print(f"loading model '{name}'")
    model = torch.hub.load("nx-ai/vision-lstm", name)
    model = model.to(device)

    # precision
    if precision == "fp32":
        autocast_ctx = NoopContext()
    elif precision == "fp16":
        autocast_ctx = torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16)
    elif precision == "bf16":
        autocast_ctx = torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16)
    else:
        raise NotImplementedError

    # iterate over dataset
    print(f"batch_size: {batch_size}")
    print(f"num_workers: {num_workers}")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    num_correct = 0
    for x, y in tqdm(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx:
            y_hat = model(x).argmax(dim=1)
        num_correct += (y_hat == y).sum()
    accuracy = num_correct / len(dataset)
    print(f"accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main(**parse_args())
