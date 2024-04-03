import pipe
import torch
from torch import nn
from torch.functional import F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


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
        self._structured_modules = {}

    def _construct_branch_structure(self):
        branches = [_ for _ in self.named_modules()]
        branches = list(
            branches
            | pipe.filter(lambda module_information: "." not in module_information[
                0] and "branch" in module_information[0])
            | pipe.sort(key=lambda module_information: float(
                module_information[0].split("branch")[1].replace("_", ".")
            ))
        )
        previous_module_name = None
        for module_name, module in branches:
            if module_name.startswith("sub"):
                if previous_module_name is None:
                    raise ValueError(f"Branch module must be defined "
                                     f"before sub-branch module")
                if previous_module_name not in module_name:
                    raise ValueError(f"Sub-branch module {module_name} is not a "
                                     f"sub-branch of precedent branch "
                                     f"{previous_module_name}")
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
    def __init__(self, in_features, out_features):
        super(GridReduction, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels=in_features, out_channels=out_features,
                        kernel_size=3, stride=2, padding=0)
        )

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self._construct_branch_structure()


class InceptionModuleF5(InceptionModuleBase):
    def __init__(self, in_features, out_features):
        super().__init__()
        if type(out_features) is not list or len(out_features) != 4:
            raise ValueError("out_features must be a list of 4 elements")

        self.branch1 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[0],
                        kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_channels=out_features[0], out_channels=out_features[0],
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=out_features[0], out_channels=out_features[0],
                        kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[1],
                        kernel_size=1, stride=1),
            BasicConv2d(in_channels=out_features[1], out_channels=out_features[1],
                        kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = InceptionBranch(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=in_features, out_channels=out_features[2],
                        kernel_size=1, stride=1, padding=0)
        )
        self.branch4 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[3],
                        kernel_size=1, stride=1, padding=0)
        )
        self._construct_branch_structure()


class InceptionModuleF6(InceptionModuleBase):
    """
    equivalent to torchvision.models.inception.InceptionC
    """

    def __init__(self, in_features, out_features, n=7):
        super().__init__()
        if type(out_features) is not list or len(out_features) != 4:
            raise ValueError("out_features must be a list of 4 elements")

        self.branch1 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[0],
                        kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_channels=out_features[0], out_channels=out_features[0],
                        kernel_size=(1, n), stride=1, padding=(0, n // 2)),
            BasicConv2d(in_channels=out_features[0], out_channels=out_features[0],
                        kernel_size=(n, 1), stride=1, padding=(n // 2, 0)),
            BasicConv2d(in_channels=out_features[0], out_channels=out_features[0],
                        kernel_size=(1, n), stride=1, padding=(0, n // 2)),
            BasicConv2d(in_channels=out_features[0], out_channels=out_features[0],
                        kernel_size=(n, 1), stride=1, padding=(n // 2, 0)),
        )
        self.branch2 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[1],
                        kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_channels=out_features[1], out_channels=out_features[1],
                        kernel_size=(1, n), stride=1, padding=(0, n // 2)),
            BasicConv2d(in_channels=out_features[1], out_channels=out_features[1],
                        kernel_size=(n, 1), stride=1, padding=(n // 2, 0)),
        )
        self.branch3 = InceptionBranch(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=in_features, out_channels=out_features[2],
                        kernel_size=1, stride=1, padding=0)
        )
        self.branch4 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[3],
                        kernel_size=1, stride=1, padding=0)
        )
        self._construct_branch_structure()


class InceptionModuleF7(InceptionModuleBase):
    def __init__(self, in_features, out_features):
        super().__init__()
        if type(out_features) is not list or len(out_features) != 6:
            raise ValueError("out_features must be a list of 4 elements")

        self.branch1 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[0] // 4,
                        kernel_size=1, stride=1, padding=0),
            BasicConv2d(in_channels=out_features[0] // 4,
                        out_channels=out_features[0] // 4,
                        kernel_size=3, stride=1, padding=1)
        )
        self.subbranch1_1 = nn.Sequential(
            BasicConv2d(in_channels=out_features[0] // 4, out_channels=out_features[0],
                        kernel_size=(1, 3), stride=1, padding=(0, 3 // 2))
        )
        self.subbranch1_2 = nn.Sequential(
            BasicConv2d(in_channels=out_features[0] // 4, out_channels=out_features[1],
                        kernel_size=(3, 1), stride=1, padding=(3 // 2, 0))
        )
        self.branch2 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[2] // 4,
                        kernel_size=1, stride=1, padding=0)
        )
        self.subbranch2_1 = nn.Sequential(
            BasicConv2d(in_channels=out_features[2] // 4, out_channels=out_features[2],
                        kernel_size=(1, 3), stride=1, padding=(0, 3 // 2))
        )
        self.subbranch2_2 = nn.Sequential(
            BasicConv2d(in_channels=out_features[2] // 4, out_channels=out_features[3],
                        kernel_size=(3, 1), stride=1, padding=(3 // 2, 0))
        )
        self.branch3 = InceptionBranch(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels=in_features, out_channels=out_features[4],
                        kernel_size=1, stride=1, padding=0)
        )
        self.branch4 = InceptionBranch(
            BasicConv2d(in_channels=in_features, out_channels=out_features[5],
                        kernel_size=1, stride=1, padding=0)
        )
        self._construct_branch_structure()


class InceptionAux(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, n_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x
