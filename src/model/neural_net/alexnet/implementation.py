import numpy as np
import torch
import torchvision.models as torch_models
from torch import nn
from utils.environment import get_env


class AlexNet(torch_models.AlexNet):
    def __init__(self, n_classes: int, image_shape: tuple[int, int], in_channels: int = 3, pretrained: bool = True, device: str = "cpu"):
        super().__init__(num_classes=n_classes)
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.device = device
        self.in_channels = in_channels

        if in_channels != self.features[0].in_channels:
            self.features[0] = nn.Conv2d(
                in_channels, self.features[0].out_channels,
                self.features[0].kernel_size, self.features[0].stride, self.features[0].padding,
                self.features[0].dilation, self.features[0].groups, self.features[0].bias is not None,
                self.features[0].padding_mode, dtype=torch.float
            )

        if pretrained:
            pretrained_weights = torch_models.AlexNet_Weights.IMAGENET1K_V1.get_state_dict(
                progress=True, model_dir=get_env("PRETRAINED_MODELS_LOCATION")
            )
            if n_classes != pretrained_weights["classifier.6.weight"].shape[0]:
                pretrained_weights["classifier.6.weight"] = pretrained_weights["classifier.6.weight"][
                                                  :n_classes]
                pretrained_weights["classifier.6.bias"] = pretrained_weights["classifier.6.bias"][
                                                :n_classes]

            if in_channels != pretrained_weights["features.0.weight"].shape[1]:
                conv1_w = pretrained_weights.pop("features.0.weight")
                conv1_w = torch.sum(conv1_w, dim=1, keepdim=True)
                conv1_w = conv1_w.repeat(1, in_channels, 1, 1)
                pretrained_weights["features.0.weight"] = conv1_w

            if self.feature_shape[-2:] != self.avgpool.output_size:
                self.avgpool = nn.AdaptiveAvgPool2d(self.feature_shape[-2:])
                classifier_input_shape = int(np.prod(self.feature_shape))
                self.classifier[1] = nn.Linear(classifier_input_shape, 4096)
                pretrained_weights["classifier.1.weight"] = pretrained_weights["classifier.1.weight"][:,
                                                  :classifier_input_shape]

            self.load_state_dict(pretrained_weights)

        self.to(device)

    @property
    def feature_shape(self) -> torch.Size:
        x = torch.randn(1, self.in_channels, *self.image_shape)
        out = self.features(x)

        return out.shape


class ModifiedAlexNet(AlexNet):
    def __init__(self, n_classes, image_shape, n_color_channels=3):
        super().__init__(n_classes=n_classes, image_shape=image_shape, in_channels=n_color_channels)
        # replace ReLU with Softmax
        self.fc1[-1] = nn.Softmax(dim=1)


if __name__ == "__main__":
    from icecream import ic
    from torchinfo import summary

    image_shape_ = (116, 116)
    in_channels = 1
    alex_net = AlexNet(n_classes=3, image_shape=image_shape_, pretrained=True, in_channels=in_channels)
    ic(
        summary(
            alex_net,
            input_size=(1, in_channels, *image_shape_),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )

    # modified_alex_net = ModifiedAlexNet(n_classes=3)
    # ic(
    #     summary(
    #         modified_alex_net,
    #         input_size=(1, 3, 299, 299),
    #         verbose=0,
    #         col_names=("kernel_size", "input_size", "output_size", "num_params"),
    #     )
    # )
