import torch
from torch import nn

from model import AlexNetFeatures, InceptionFeatures


class D2(nn.Module):
    def __init__(self, n_classes, n_color_channels=3):
        super().__init__()
        self.alex_net_features = AlexNetFeatures(n_color_channels=n_color_channels)
        self.inception_v2_features = InceptionFeatures(
            n_classes=n_classes, n_color_channels=n_color_channels
        )

        self.fc1 = nn.Sequential(
            nn.Linear(
                self.alex_net_features.out_channels * 8 * 8
                + self.inception_v2_features.out_channels * 8 * 8,
                64,
            ),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.classification = nn.Sequential(nn.Linear(64, n_classes), nn.Softmax(dim=1))

    def forward(self, x):
        features_alex_net = self.alex_net_features(x)
        features_alex_net = torch.flatten(features_alex_net, 1)

        features_inception_v2 = self.inception_v2_features(x)
        if type(features_inception_v2) is not torch.Tensor:
            features_inception_v2 = features_inception_v2[0]
        features_inception_v2 = torch.flatten(features_inception_v2, 1)

        features = torch.cat([features_alex_net, features_inception_v2], dim=1)

        out = self.fc1(features)
        out = self.fc2(out)
        out = self.classification(out)

        return out


if __name__ == "__main__":
    from icecream import ic
    from torchinfo import summary

    d2 = D2(n_classes=3)
    ic(
        summary(
            d2,
            input_size=(1, 3, 299, 299),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )
