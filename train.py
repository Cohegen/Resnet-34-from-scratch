import torch
import torch.nn as nn
import torch.optim as optim

from data_setup import create_dataloaders
from engine import train
from model import Resnet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = create_dataloaders(batch_size=64)
    model = Resnet(num_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=5,
        device=device,
    )

    return results


if __name__ == "__main__":
    main()
