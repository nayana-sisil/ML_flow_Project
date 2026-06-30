import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN

import mlflow
import mlflow.pytorch


mlflow.set_experiment("cifar10-cnn-cpu")


device = torch.device("cpu")


BASE_DIR = r"C:/Users/Ewis/Downloads/archive (5)/cifar10"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = datasets.ImageFolder(
    root=os.path.join(BASE_DIR, "train"), transform=transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(BASE_DIR, "test"), transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Classes:", train_dataset.classes)


model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# Accuracy function
def accuracy(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


epochs = 3


with mlflow.start_run():
    mlflow.log_param("model", "CNN")
    mlflow.log_param("optimizer", "SGD")
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", epochs)

    for epoch in range(epochs):
        print(f"\n===== START EPOCH {epoch + 1}/{epochs} =====")

        model.train()
        running_loss = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=float(loss.item()))

        # Evaluation

        train_acc = accuracy(train_loader)
        test_acc = accuracy(test_loader)

        avg_loss = running_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1} Summary:", flush=True)
        print(f"Loss: {avg_loss:.4f}", flush=True)
        print(f"Train Acc: {train_acc:.4f}", flush=True)
        print(f"Test Acc: {test_acc:.4f}", flush=True)

        # Log metrics to MLflow

        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("test_acc", test_acc, step=epoch)

    # Save model locally

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn.pth")

    print("\nModel saved to models/cnn.pth")

    # Log model to MLflow
    example_input = torch.randn(1, 3, 32, 32)

import torch

example_input = torch.randn(1, 3, 32, 32)

mlflow.pytorch.log_model(
    model, "model", input_example=example_input, serialization_format="pickle"
)
