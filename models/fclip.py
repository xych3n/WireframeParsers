from typing import Sequence, Tuple
from functools import partial

import torch
from torch import Tensor, nn

from .backbone.hourglass import Bottleneck, StackedHourglass
from .lcnn import MultitaskHead, propose_junctions


def structrual_nms(lines: Tensor, scores: Tensor, threshold: float = 2):
    lines = lines.reshape(-1, 2, 2)
    euid = lambda x, y: ((x - y) ** 2).sum(axis=-1)
    dist = torch.minimum(
        euid(lines[:, None,  0], lines[None, :, 0]) + euid(lines[:, None, 1], lines[None, :, 1]),
        euid(lines[:, None,  1], lines[None, :, 0]) + euid(lines[:, None, 0], lines[None, :, 1])
    )
    indices = dist <= threshold
    diagonal = torch.eye(len(lines), dtype=bool, device=lines.device)
    indices[diagonal] = False
    drop = indices[0]
    for i in range(1, len(lines) - 2):
        if drop[i]:
            continue
        drop[i+1:] |= indices[i, i+1:]
    lines, scores = lines[~drop], scores[~drop]
    lines = lines.reshape(-1, 4)
    return lines, scores


class LineBlock(nn.Module):
    def __init__(self, planes: int) -> None:
        super().__init__()

        self.conv_h = nn.Conv2d(planes, planes, kernel_size=(7, 1), padding=(3, 0))
        self.conv_v = nn.Conv2d(planes, planes, kernel_size=(1, 7), padding=(0, 3))

    def forward(self, x: Tensor):
        x = torch.maximum(self.conv_h(x), self.conv_v(x))
        return x


class BottleneckLine(Bottleneck):
    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__(inplanes, planes, stride)

        self.conv2 = LineBlock(planes)


class StackedHourglassLine(StackedHourglass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for hourglass in self.hourglasses:
            hourglass += nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )


def build_backbone(task_sizes: Sequence[int], line_block: bool):
    if line_block:
        return StackedHourglassLine(
            BottleneckLine,
            partial(MultitaskHead, task_sizes=task_sizes),
            num_classes=sum(task_sizes),
        )
    else:
        return StackedHourglass(
            Bottleneck,
            partial(MultitaskHead, task_sizes=task_sizes),
            num_classes=sum(task_sizes),
        )


class FClip(nn.Module):
    def __init__(self, line_block: bool = False) -> None:
        super().__init__()

        self.backbone = build_backbone([2, 2, 1, 1], line_block)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        assert not self.training and len(images) == 1

        _, heatmaps_list = self.backbone(images)
        heatmaps = heatmaps_list[0]
        cloc, = heatmaps[:, 0:2].softmax(dim=1)[:, [1]]
        coff, = heatmaps[:, 2:4].sigmoid()[:, [1, 0]]   # assure [dx, dy] format
        llen = heatmaps[:, 4:5].sigmoid()
        lang = heatmaps[:, 5:6].sigmoid()

        centers, scores, indices = propose_junctions(cloc, coff,
            k=1000, soft=0.8, return_indices=True)
        radii = llen.flatten()[indices] * 64
        angles = lang.flatten()[indices] * torch.pi
        displs = torch.stack((angles.cos(), -angles.sin().abs())) * radii
        lines = torch.cat((centers + displs.t(), centers - displs.t()), dim=1)

        lines, scores = structrual_nms(lines, scores)

        return lines, scores
