from typing import Callable, Optional, Type

from torch import Tensor, nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    """Pre-activation Bottleneck module"""
    expansion: int = 2

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1),
                # NOTE Why there's no norm layer?
                # nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.downsample(x) if self.downsample else x
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))
        x = self.conv3(self.relu(self.bn3(x)))
        x += residual
        return x


class Hourglass(nn.Module):
    def __init__(self, block: Type[Bottleneck], planes: int, blocks: int = 1, depth: int = 4) -> None:
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(*[
                    block(planes * block.expansion, planes)
                    for _ in range(blocks)
                ]) for _ in range(4 if level == 0 else 3)
            ]) for level in range(depth)
        ])

    def _forward_recursive(self, x: Tensor, level: int) -> Tensor:
        residual = self.layers[level][0](x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.layers[level][1](x)
        if level > 0:
            x = self._forward_recursive(x, level - 1)
        else:
            x = self.layers[level][3](x)
        x = self.layers[level][2](x)
        x = F.interpolate(x, scale_factor=2)
        x += residual
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_recursive(x, self.depth - 1)


class StackedHourglass(nn.Module):
    def __init__(
        self,
        block: Type[Bottleneck],
        head: Optional[Callable[[int], nn.Module]] = None,
        num_classes: int = 0,
        blocks: int = 1,
        depth: int = 4,
        num_stacks: int = 2,
    ) -> None:
        super().__init__()

        # ResNet layers
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.layer1 = self._make_layer(Bottleneck, 64, 1)
        self.layer2 = self._make_layer(Bottleneck, 128, 1)
        self.layer3 = self._make_layer(Bottleneck, 128, 1)
        # Hourglass modules
        inplanes = 128 * block.expansion
        self.hourglasses = nn.ModuleList([
            nn.Sequential(
                Hourglass(block, 128, blocks, depth),
                # NOTE Why there're these additional layers designed?
                self._make_layer(block, 128, blocks),
                nn.Conv2d(inplanes, inplanes, kernel_size=1),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
            ) for _ in range(num_stacks)
        ])
        self.heads = None
        if head is not None:
            assert num_classes != 0
            self.heads = nn.ModuleList([
                head(inplanes)
                for _ in range(num_stacks)
            ])
            # Intermedia supervision layers
            self.refinements = nn.ModuleList([
                nn.Conv2d(inplanes, inplanes, kernel_size=1)
                for _ in range(num_stacks - 1)
            ])
            self.remap_convs = nn.ModuleList([
                nn.Conv2d(num_classes, inplanes, kernel_size=1)
                for _ in range(num_stacks - 1)
            ])

    def _make_layer(self, block: Type[Bottleneck], planes: int, blocks: int, stride: int = 1):
        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        layers.extend([block(self.inplanes, planes) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        heatmaps = []
        for i in range(len(self.hourglasses)):
            residual = x
            x = self.hourglasses[i](x)
            if self.heads is not None:
                h = self.heads[i](x)
                heatmaps.insert(0, h)
            if i < len(self.hourglasses) - 1:
                if self.heads is not None:
                    # Intermedia supervision
                    x = self.refinements[i](x) + self.remap_convs[i](h)
                x += residual
        if self.heads is not None:
            return x, heatmaps
        return x
