from torch import nn
from torchinfo import summary

from model.alexnet.base import AlexNetBase


class AlexNet(AlexNetBase):
    def __init__(self, n_classes, n_color_channels=3):
        super().__init__(n_color_channels=n_color_channels)

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(nn.Linear(4096, n_classes))

    def forward(self, x):
        out = super().forward(x)

        out = self.flatten(out)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class ModifiedAlexNet(nn.Module):
    ...


if __name__ == "__main__":
    from icecream import ic

    alex_net = AlexNet(n_classes=3)
    ic(summary(alex_net, input_size=(1, 3, 224, 224), verbose=0))
