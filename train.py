from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

image, label = train_dataset[0]

print(image.shape)
print(label)


# if __name__ == "__main__":
#     main()
