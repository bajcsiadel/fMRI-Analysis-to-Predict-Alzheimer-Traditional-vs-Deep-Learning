from collections import namedtuple
from typing import Optional

import torch
from torch import nn

from model.neural_net.inception_v2.base import InceptionFeatures

InceptionOutputs = namedtuple("InceptionOutputs", ["logits", "aux_logits"])
InceptionOutputs.__annotations__ = {
    "logits": torch.Tensor,
    "aux_logits": Optional[torch.Tensor],
}


class InceptionV2(nn.Module):
    def __init__(self, n_classes, n_color_channels=3, aux_logits=True):
        super().__init__()

        self.features = InceptionFeatures(
            n_classes=n_classes,
            n_color_channels=n_color_channels,
            aux_logits=aux_logits,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(2048, n_classes)

    def forward(self, x):
        out, aux_out = self.features(x)

        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.flatten(out)
        out = self.classifier(out)

        if self.training:
            return InceptionOutputs(logits=out, aux_logits=aux_out)

        return out


class ModifiedInceptionV2(nn.Module):
    ...


if __name__ == "__main__":
    from icecream import ic
    from torchinfo import summary

    from model.neural_net.inception_v2.base import (
        GridReduction,
        InceptionModuleF5,
        InceptionModuleF6,
        InceptionModuleF7,
    )

    inception = InceptionV2(n_classes=2)
    ic(
        summary(
            inception,
            input_size=(1, 1, 116, 116),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )

    inception_f5 = InceptionModuleF5(288, [96, 64, 64, 64])
    ic(
        summary(
            inception_f5,
            input_size=(1, 288, 35, 35),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )
    inception_f6 = InceptionModuleF6(768, 192, filter_size_1x1=160)
    ic(
        summary(
            inception_f6,
            input_size=(1, 768, 17, 17),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )
    inception_f7 = InceptionModuleF7(1280, [384, 384, 384, 384, 192, 320])
    ic(
        summary(
            inception_f7,
            input_size=(1, 1280, 8, 8),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )

    grid_reduction = GridReduction(4 * 96, 384)
    ic(
        summary(
            grid_reduction,
            input_size=(1, 384, 35, 35),
            verbose=0,
            col_names=("kernel_size", "input_size", "output_size", "num_params"),
        )
    )
