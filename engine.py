import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    train_loss, train_correct, train_samples = 0.0, 0, 0
    non_blocking = device.type != "cpu"

    for X, y in dataloader:
        X = X.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        train_correct += (y_pred.argmax(dim=1) == y).sum().item()
        train_samples += y.size(0)

    train_loss /= train_samples
    train_acc = train_correct / train_samples
    return train_loss, train_acc

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):
    model.eval()
    test_loss, test_correct, test_samples = 0.0, 0, 0
    non_blocking = device.type != "cpu"

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)

            test_loss += loss.item() * y.size(0)
            test_correct += (test_pred_logits.argmax(dim=1) == y).sum().item()
            test_samples += y.size(0)

    test_loss /= test_samples
    test_acc = test_correct / test_samples
    return test_loss, test_acc

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module | None = None,
    epochs: int = 5,
    device: torch.device | str = "cpu",
):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    device = torch.device(device)
    
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
