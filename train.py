import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CNN


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
optimizer = optim.Adam(model.parameters(), lr=0.001)


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


# Training loop

epochs = 2

for epoch in range(epochs):
    model.train()
    running_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    train_acc = accuracy(train_loader)
    test_acc = accuracy(test_loader)

    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"Loss: {running_loss / len(train_loader):.4f}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")


os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/cnn.pth")
print("\nModel saved to models/cnn.pth")
