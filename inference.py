import torch
from torchvision import transforms
from PIL import Image

from model import CNN


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


device = torch.device("cpu")


model = CNN().to(device)
model.load_state_dict(torch.load("models/cnn.pth", map_location=device))
model.eval()


transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return classes[pred.item()]


if __name__ == "__main__":
    img_path = "test.jpg"

    result = predict(img_path)
    print("\n====================")
    print("Prediction:", result)
    print("====================\n")
