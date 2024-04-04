import torch
from torch import nn

from model import ModifiedAlexNet, ModifiedInceptionV2


class D2(nn.Module):
    def __init__(self, n_classes, n_color_channels=3):
        super().__init__()
        self.alexnet = ModifiedAlexNet(
            n_classes=n_classes, n_color_channels=n_color_channels
        )
        self.inception_v2 = ModifiedInceptionV2(
            n_classes=n_classes, n_color_channels=n_color_channels
        )

        self.fc1 = nn.Sequential(
            nn.Linear(123, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.classification = nn.Sequential(
            nn.Linear(64, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features_alexnet = self.alexnet(x)
        features_inception_v2 = self.inception_v2(x)

        features = torch.cat([features_alexnet, features_inception_v2], dim=1)

        out = self.fc1(features)
        out = self.fc2(out)
        out = self.classification(out)

        return out
