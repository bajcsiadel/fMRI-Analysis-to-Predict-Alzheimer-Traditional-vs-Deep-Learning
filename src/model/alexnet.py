from torch import nn

from model._base import AlexNetBase


class AlexNet(AlexNetBase):
    # https://blog.paperspace.com/alexnet-pytorch/#alexnet-from-scratch
    def __init__(self, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
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

        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    from icecream import ic

    alex_net = AlexNet(n_classes=3)
    ic(alex_net)
