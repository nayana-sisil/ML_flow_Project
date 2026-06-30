import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # BEST RESULT FOUND


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


def get_preds(loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


epochs = 1

with mlflow.start_run():
    # log params
    mlflow.log_param("model", "CNN")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("lr", 0.0005)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", epochs)

    # training loop
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

        # evaluation
        train_acc = accuracy(train_loader)
        test_acc = accuracy(test_loader)

        avg_loss = running_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f}")

        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("test_acc", test_acc, step=epoch)

    # CONFUSION MATRIX

    y_true, y_pred = get_preds(test_loader)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # SAMPLE PREDICTIONS

    images, labels = next(iter(test_loader))

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 6))

    for i in range(8):
        plt.subplot(2, 4, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 0.5) + 0.5  # unnormalize
        plt.imshow(img)
        plt.title(f"P:{preds[i].item()} T:{labels[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("samples.png")
    mlflow.log_artifact("samples.png")

    # SAVE MODEL
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn.pth")

    print("\nModel saved to models/cnn.pth")

    # LOG MODEL

    example_input = torch.randn(1, 3, 32, 32)

    mlflow.pytorch.log_model(
        model, "model", input_example=example_input, serialization_format="pickle"
    )

    print("\nMLflow run complete ")
