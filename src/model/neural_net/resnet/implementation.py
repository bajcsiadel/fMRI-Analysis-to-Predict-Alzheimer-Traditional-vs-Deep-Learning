import torch
from sklearn.base import BaseEstimator
from torch import nn, Tensor
from torchinfo import summary
from torchvision.models import resnet as torch_resnet

from utils.environment import get_env


_model_parameters = {
    18: {
        "init": {
            "block": torch_resnet.BasicBlock,
            "layers": [2, 2, 2, 2],
        },
        "weights": torch_resnet.ResNet18_Weights.IMAGENET1K_V1
    },
    20: {
        "init": {
            "block": torch_resnet.BasicBlock,
            "layers": [3, 3, 3],
        },
        "weights": None,
    },
    34: {
        "init": {
            "block": torch_resnet.BasicBlock,
            "layers": [3, 4, 6, 3],
        },
        "weights": torch_resnet.ResNet34_Weights.IMAGENET1K_V1
    },
    50: {
        "init": {
            "block": torch_resnet.Bottleneck,
            "layers": [3, 4, 6, 3],
        },
        "weights": torch_resnet.ResNet50_Weights.IMAGENET1K_V1
    },
    101: {
        "init": {
            "block": torch_resnet.Bottleneck,
            "layers": [3, 4, 23, 3],
        },
        "weights": torch_resnet.ResNet101_Weights.IMAGENET1K_V1
    },
    152: {
        "init": {
            "block": torch_resnet.Bottleneck,
            "layers": [3, 8, 36, 3],
        },
        "weights": torch_resnet.ResNet152_Weights.IMAGENET1K_V1
    },
}


class ResNet(BaseEstimator, torch_resnet.ResNet):
    def __init__(self, size: int, n_classes: int, in_channels: int = 3, pretrained: bool = True, device: str = "cpu"):
        super(ResNet, self).__init__(num_classes=n_classes, **_model_parameters[size]["init"])

        self.size = size
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.device = device
        self.in_channels = in_channels

        if in_channels != self.conv1.in_channels:
            self.conv1 = nn.Conv2d(
                in_channels, self.conv1.out_channels,
                self.conv1.kernel_size, self.conv1.stride, self.conv1.padding,
                self.conv1.dilation, self.conv1.groups, self.conv1.bias,
                self.conv1.padding_mode, dtype=torch.float
            )

        if pretrained and _model_parameters[size]["weights"] is not None:
            pretrained_weights = _model_parameters[size]["weights"].get_state_dict(
                progress=True, model_dir=get_env("PRETRAINED_MODELS_LOCATION")
            )
            if n_classes != pretrained_weights["fc.weight"].shape[0]:
                pretrained_weights["fc.weight"] = pretrained_weights["fc.weight"][
                                                  :n_classes]
                pretrained_weights["fc.bias"] = pretrained_weights["fc.bias"][
                                                :n_classes]

            if in_channels != pretrained_weights["conv1.weight"].shape[1]:
                conv1_w = pretrained_weights.pop("conv1.weight")
                conv1_w = torch.sum(conv1_w, dim=1, keepdim=True)
                conv1_w = conv1_w.repeat(1, in_channels, 1, 1)
                pretrained_weights["conv1.weight"] = conv1_w

            self.load_state_dict(pretrained_weights)

        self.to(device)

    def forward(self, x: Tensor, apply_softmax: bool = False) -> Tensor:
        x = super(ResNet, self).forward(x)
        if apply_softmax:
            x = self.softmax(x)
        return x


if __name__ == "__main__":
    from icecream import ic

    m = ResNet(18, 1, 3, True)
    ic(
        summary(
            m,
            input_size=(1, 3, 112, 112),
            col_names=("input_size", "output_size", "kernel_size"),
            depth=4,
        )
    )