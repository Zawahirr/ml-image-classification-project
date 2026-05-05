import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        channels, height, width = input_shape
        flattened_size = channels * height * width

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        channels, height, width = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            flattened_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LeNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        channels, height, width = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            flattened_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class CustomCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        channels, height, width = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            flattened_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name, input_shape, num_classes):
    model_key = model_name.lower().replace("-", "_")

    if model_key == "mlp":
        return MLP(input_shape=input_shape, num_classes=num_classes)
    if model_key in {"simple_cnn", "simplecnn"}:
        return SimpleCNN(input_shape=input_shape, num_classes=num_classes)
    if model_key == "lenet":
        return LeNet(input_shape=input_shape, num_classes=num_classes)
    if model_key in {"custom_cnn", "improved_cnn"}:
        return CustomCNN(input_shape=input_shape, num_classes=num_classes)

    raise ValueError("model_name must be 'mlp', 'simple_cnn', 'lenet', or 'custom_cnn'")
