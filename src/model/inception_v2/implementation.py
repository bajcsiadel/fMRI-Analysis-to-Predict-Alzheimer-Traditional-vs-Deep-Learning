import torch
import torchvision.models
from torch import nn
from torchinfo import summary

from model.inception_v2.base import InceptionModuleF7, InceptionModuleF5, \
    InceptionModuleF6, BasicConv2d, GridReduction


class AuxClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(AuxClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=128, kernel_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 128, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        N = x.shape[0]
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        return x


class InceptionV2(nn.Module):
    def __init__(self, n_color_channel, n_classes):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConv2d(n_color_channel, 32, kernel_size=3, stride=2, padding=0),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=0),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64, 80, kernel_size=3, stride=1, padding=0),
            BasicConv2d(80, 192, kernel_size=3, stride=2, padding=0),
            BasicConv2d(192, 288, kernel_size=3, stride=1, padding=1)
        )

        self.inception_module_1 = nn.Sequential(
            InceptionModuleF5(in_features=288, out_features=[96, 96, 96, 96]),
            InceptionModuleF5(in_features=4 * 96, out_features=[96, 96, 96, 96]),
            InceptionModuleF5(in_features=4 * 96, out_features=[96, 96, 96, 96]),
        )

        self.grid_reduction_1 = GridReduction(4 * 96, 384)
        self.aux_classifier = AuxClassifier(768, n_classes)

        self.inception_module_2 = nn.Sequential(
            InceptionModuleF6(in_features=768, out_features=[160, 160, 160, 160]),
            InceptionModuleF6(in_features=4 * 160, out_features=[160, 160, 160, 160]),
            InceptionModuleF6(in_features=4 * 160, out_features=[160, 160, 160, 160]),
            InceptionModuleF6(in_features=4 * 160, out_features=[160, 160, 160, 160]),
            InceptionModuleF6(in_features=4 * 160, out_features=[160, 160, 160, 160]),
        )
        self.grid_reduction_2 = GridReduction(4 * 160, 640)

        self.inception_module_3 = nn.Sequential(
            InceptionModuleF7(in_features=1280, out_features=[256, 256, 192, 192, 64, 64]),
            InceptionModuleF7(in_features=1024, out_features=[384, 384, 384, 384, 256, 256]),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, n_classes),
            nn.Softmax()
        )

    def forward(self, x):
        N = x.shape[0]
        out = self.stem(x)

        out = self.inception_module_1(out)
        out = self.grid_reduction_1(out)

        aux_out = self.aux_classifier(out)

        out = self.inception_module_2(out)
        out = self.grid_reduction_2(out)

        out = self.inception_module_3(out)

        out = self.avg_pool(out)
        out = out.reshape(N, -1)
        out = self.classifier(out)

        if self.training:
            return out, aux_out

        return out


class ModifiedInceptionV2(nn.Module):
    ...


if __name__ == "__main__":
    from icecream import ic
    inception = GridReduction(4 * 96, 384)
    ic(summary(inception, input_size=(1, 384, 35, 35)))
