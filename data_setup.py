import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def create_dataloaders(batch_size: int = 32, num_workers: int = 8):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.FashionMNIST(
        root="train",
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.FashionMNIST(
        root="test",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader
