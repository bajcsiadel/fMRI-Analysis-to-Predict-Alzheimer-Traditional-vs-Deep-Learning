import copy
from collections import OrderedDict

import pipe
import torch
from torch import nn
from torch.functional import F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class RepeatedModule(nn.Sequential):
    def __init__(self, module, n_repeats):
        super().__init__()
        for _ in range(n_repeats):
            self.add_module(f"{module.__class__.__name__}_{_}", copy.deepcopy(module))

        self.in_channels = module.in_channels
        self.out_channels = module.out_channels


class InceptionBranch(nn.Sequential):
    def __init__(self, *modules, sub_branches=None):
        super().__init__(*modules)
        if sub_branches is None:
            self._sub_branches = []
        else:
            self._sub_branches = sub_branches

    def add_sub_branch(self, module):
        self._sub_branches.append(module)

    def forward(self, x):
        outs = []
        out_1 = super().forward(x)
        if len(self._sub_branches) > 0:
            for sub_branch in self._sub_branches:
                outs.append(sub_branch(out_1))
        else:
            outs.append(out_1)

        return torch.cat(outs, dim=1)


class InceptionModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self._structured_modules = OrderedDict()

    def _construct_branch_structure(self):
        branches = [_ for _ in self.named_modules()]
        branches = list(
            branches
            | pipe.filter(
                lambda module_information: "." not in module_information[0]
                and "branch" in module_information[0]
            )
            | pipe.sort(
                key=lambda module_information: float(
                    module_information[0].split("branch")[1].replace("_", ".")
                )
            )
        )
        previous_module_name = None
        for module_name, module in branches:
            if module_name.startswith("sub"):
                if previous_module_name is None:
                    raise ValueError(
                        f"Branch module must be defined " f"before sub-branch module"
                    )
                if previous_module_name not in module_name:
                    raise ValueError(
                        f"Sub-branch module {module_name} is not a "
                        f"sub-branch of precedent branch "
                        f"{previous_module_name}"
                    )
                self._structured_modules[previous_module_name].add_sub_branch(module)
            else:
                self._structured_modules[module_name] = module
                previous_module_name = module_name

    def forward(self, x):
        outs = []
        for branch in self._structured_modules.values():
            outs.append(branch(x))

        return torch.cat(outs, dim=1)


class GridReduction(InceptionModuleBase):
    def __init__(self, in_features, out_features, filter_size_1x1=None):
        super(GridReduction, self).__init__()
        if filter_size_1x1 is None:
            filter_size_1x1 = out_features
        if type(out_features) is int:
            out_features = [out_features] * 2
        if type(out_features) is not list or len(out_features) != 2:
            raise ValueError("out_features must be a list of 2 elements")

        self.branch1 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=filter_size_1x1,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BasicConv2d(
                in_channels=filter_size_1x1,
                out_channels=out_features[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BasicConv2d(
                in_channels=out_features[0],
                out_channels=out_features[0],
                kernel_size=3,
                stride=2,
                padding=0,
            ),
        )
        self.branch2 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=filter_size_1x1,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BasicConv2d(
                in_channels=filter_size_1x1,
                out_channels=out_features[1],
                kernel_size=3,
                stride=2,
                padding=0,
            ),
        )
        self.branch3 = InceptionBranch(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.in_channels = in_features
        self.out_channels = sum(out_features) + in_features

        self._construct_branch_structure()


class InceptionModuleF5(InceptionModuleBase):
    def __init__(self, in_features, out_features):
        super().__init__()
        if type(out_features) is int:
            out_features = [out_features] * 4
        if type(out_features) is not list or len(out_features) != 4:
            raise ValueError("out_features must be a list of 4 elements")

        self.branch1 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[0] * 2 // 3,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BasicConv2d(
                in_channels=out_features[0] * 2 // 3,
                out_channels=out_features[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            BasicConv2d(
                in_channels=out_features[0],
                out_channels=out_features[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.branch2 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[1] * 3 // 4,
                kernel_size=1,
                stride=1,
            ),
            BasicConv2d(
                in_channels=out_features[1] * 3 // 4,
                out_channels=out_features[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.branch3 = InceptionBranch(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.branch4 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.in_channels = in_features
        self.out_channels = sum(out_features)

        self._construct_branch_structure()


class InceptionModuleF6(InceptionModuleBase):
    """
    equivalent to torchvision.models.inception.InceptionC
    """

    def __init__(self, in_features, out_features, n=7, filter_size_1x1=None):
        super().__init__()
        if type(out_features) is int:
            out_features = [out_features] * 4
        if type(out_features) is not list or len(out_features) != 4:
            raise ValueError("out_features must be a list of 4 elements")

        if filter_size_1x1 is None:
            filter_size_1x1 = out_features

        self.branch1 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=filter_size_1x1,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BasicConv2d(
                in_channels=filter_size_1x1,
                out_channels=filter_size_1x1,
                kernel_size=(1, n),
                stride=1,
                padding=(0, n // 2),
            ),
            BasicConv2d(
                in_channels=filter_size_1x1,
                out_channels=filter_size_1x1,
                kernel_size=(n, 1),
                stride=1,
                padding=(n // 2, 0),
            ),
            BasicConv2d(
                in_channels=filter_size_1x1,
                out_channels=filter_size_1x1,
                kernel_size=(1, n),
                stride=1,
                padding=(0, n // 2),
            ),
            BasicConv2d(
                in_channels=filter_size_1x1,
                out_channels=out_features[0],
                kernel_size=(n, 1),
                stride=1,
                padding=(n // 2, 0),
            ),
        )
        self.branch2 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=filter_size_1x1,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BasicConv2d(
                in_channels=filter_size_1x1,
                out_channels=filter_size_1x1,
                kernel_size=(1, n),
                stride=1,
                padding=(0, n // 2),
            ),
            BasicConv2d(
                in_channels=filter_size_1x1,
                out_channels=out_features[1],
                kernel_size=(n, 1),
                stride=1,
                padding=(n // 2, 0),
            ),
        )
        self.branch3 = InceptionBranch(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.branch4 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.in_channels = in_features
        self.out_channels = sum(out_features)

        self._construct_branch_structure()


class InceptionModuleF7(InceptionModuleBase):
    def __init__(self, in_features, out_features):
        super().__init__()
        if type(out_features) is int:
            out_features = [out_features] * 6
        if type(out_features) is not list or len(out_features) != 6:
            raise ValueError("out_features must be a list of 4 elements")

        self.branch1 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=448,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            BasicConv2d(
                in_channels=448,
                out_channels=out_features[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.subbranch1_1 = nn.Sequential(
            InceptionBranch(
                BasicConv2d(
                    in_channels=out_features[0],
                    out_channels=out_features[0],
                    kernel_size=(1, 3),
                    stride=1,
                    padding=(0, 3 // 2),
                )
            )
        )
        self.subbranch1_2 = nn.Sequential(
            InceptionBranch(
                BasicConv2d(
                    in_channels=out_features[0],
                    out_channels=out_features[1],
                    kernel_size=(3, 1),
                    stride=1,
                    padding=(3 // 2, 0),
                )
            )
        )
        self.branch2 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.subbranch2_1 = nn.Sequential(
            InceptionBranch(
                BasicConv2d(
                    in_channels=out_features[2],
                    out_channels=out_features[2],
                    kernel_size=(1, 3),
                    stride=1,
                    padding=(0, 3 // 2),
                )
            )
        )
        self.subbranch2_2 = nn.Sequential(
            InceptionBranch(
                BasicConv2d(
                    in_channels=out_features[2],
                    out_channels=out_features[3],
                    kernel_size=(3, 1),
                    stride=1,
                    padding=(3 // 2, 0),
                )
            )
        )
        self.branch3 = InceptionBranch(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[4],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.branch4 = InceptionBranch(
            BasicConv2d(
                in_channels=in_features,
                out_channels=out_features[5],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.in_channels = in_features
        self.out_channels = sum(out_features)

        self._construct_branch_structure()


class InceptionAux(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv2 = BasicConv2d(128, 768, kernel_size=5)
        self.conv2.stddev = 0.01
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(768, n_classes)
        self.fc.stddev = 0.001

        self.in_channels = in_channels
        self.out_channels = n_classes

    def forward(self, x):
        # N x 768 x 17 x 17
        out = self.pool1(x)
        # N x 768 x 5 x 5
        out = self.conv1(out)
        # N x 128 x 5 x 5
        out = self.conv2(out)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        out = self.pool2(out)
        # N x 768 x 1 x 1
        out = self.flatten(out)
        # N x 768
        out = self.fc(out)
        # N x n_classes
        return out


class InceptionFeatures(nn.Module):
    def __init__(self, n_classes, n_color_channels=3, aux_logits=True):
        super().__init__()

        self.stem = nn.Sequential(
            BasicConv2d(n_color_channels, 32, kernel_size=3, stride=2, padding=0),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=0),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64, 80, kernel_size=3, stride=1, padding=0),
            BasicConv2d(80, 192, kernel_size=3, stride=2, padding=0),
            BasicConv2d(192, 288, kernel_size=3, stride=1, padding=1),
        )

        self.inception_module_1 = RepeatedModule(
            InceptionModuleF5(in_features=288, out_features=[96, 64, 64, 64]),
            n_repeats=3,
        )

        self.grid_reduction_1 = GridReduction(288, [178, 302], 64)

        self.inception_module_2 = nn.Sequential(
            InceptionModuleF6(in_features=768, out_features=192, filter_size_1x1=128),
            InceptionModuleF6(in_features=768, out_features=192, filter_size_1x1=160),
            InceptionModuleF6(in_features=768, out_features=192, filter_size_1x1=160),
            InceptionModuleF6(in_features=768, out_features=192, filter_size_1x1=160),
            InceptionModuleF6(in_features=768, out_features=192, filter_size_1x1=192),
        )

        if aux_logits:
            self.aux_classifier = InceptionAux(768, n_classes)

        self.grid_reduction_2 = GridReduction(768, [194, 318], 192)

        self.inception_module_3 = nn.Sequential(
            InceptionModuleF7(
                in_features=1280, out_features=[384, 384, 384, 384, 192, 320]
            ),
            InceptionModuleF7(
                in_features=2048, out_features=[384, 384, 384, 384, 192, 320]
            ),
        )

        self.in_channels = n_color_channels
        self.out_channels = 2048

    def forward(self, x):
        out = self.stem(x)

        out = self.inception_module_1(out)
        out = self.grid_reduction_1(out)

        out = self.inception_module_2(out)

        aux_out = None
        if self.aux_classifier is not None:
            aux_out = self.aux_classifier(out)

        out = self.grid_reduction_2(out)

        out = self.inception_module_3(out)

        return out, aux_out
