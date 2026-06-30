from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(
    root="C:/Users/Ewis/Downloads/archive (5)/cifar10/train", transform=transform
)

test_dataset = datasets.ImageFolder(
    root="C:/Users/Ewis/Downloads/archive (5)/cifar10/test", transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Classes:", train_dataset.classes)
print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))
