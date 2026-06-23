import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def create_dataloaders(batch_size: int = 32, num_workers: int | None = None):
    if num_workers is None:
        num_workers = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    return train_loader, test_loader
