from torch import nn

from model.alexnet.base import AlexNetFeatures


class AlexNet(nn.Module):
    def __init__(self, n_classes, n_color_channels=3):
        super().__init__()

        self.features = AlexNetFeatures(n_color_channels=n_color_channels)

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.features.out_channels * 8 * 8, 4096),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, n_classes),
            nn.Softmax(),
        )

        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.features.layer2[0].bias, 1)
        nn.init.constant_(self.features.layer4[0].bias, 1)
        nn.init.constant_(self.features.layer5[0].bias, 1)

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class ModifiedAlexNet(AlexNet):
    def __init__(self, n_classes, n_color_channels=3):
        super().__init__(n_classes=n_classes, n_color_channels=n_color_channels)
        # replace ReLU with Softmax
        self.fc1[-1] = nn.Softmax(dim=1)


if __name__ == "__main__":
    from icecream import ic
    from torchinfo import summary

    alex_net = AlexNet(n_classes=3)
    ic(
        summary(
            alex_net,
            input_size=(1, 3, 227, 227),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )

    modified_alex_net = ModifiedAlexNet(n_classes=3)
    ic(
        summary(
            modified_alex_net,
            input_size=(1, 3, 299, 299),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )
