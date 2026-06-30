from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


train_dataset = datasets.CIFAR10(
    root="data", train=True, download=False, transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data", train=False, download=False, transform=transform
)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))
