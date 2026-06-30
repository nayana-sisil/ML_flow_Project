import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils import CIFAR10_CLASSES


class CIFAR10Predictor:
    def __init__(self, run_id=None, model_uri=None):
        if model_uri:
            uri = model_uri
        elif run_id:
            uri = f"runs:/{run_id}/model"
        else:
            raise ValueError("Provide either run_id or model_uri")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = mlflow.pytorch.load_model(uri).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
        return {
            "class": CIFAR10_CLASSES[pred_idx],
            "class_id": pred_idx,
            "probabilities": probs.squeeze().tolist(),
        }
