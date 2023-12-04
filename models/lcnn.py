from functools import partial
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone.hourglass import Bottleneck, StackedHourglass


def nms(x: Tensor, kernel_size: int = 3, soft: float = 0) -> Tensor:
    mask = x == F.max_pool2d(x, kernel_size=kernel_size,
                             stride=1, padding=kernel_size//2)
    return x * (mask + ~mask * soft)


def propose_junctions(jloc: Tensor, joff: Tensor, k: int = 0, threshold: float = 0,
                      kernel_size: int = 3, soft: float = 0, return_indices: bool = False):
    assert jloc.ndim == 3 and jloc.size(0) == 1
    assert joff.ndim == 3 and joff.size(0) == 2
    _, H, W = jloc.shape
    jloc = nms(jloc, kernel_size, soft)
    jloc = jloc.flatten()
    joff = joff.flatten(start_dim=1)
    if k > 0:
        scores, indices = torch.topk(jloc, k)
    else:
        indices = (jloc > threshold).nonzero()
        scores = jloc[indices]
    y = indices // W + joff[1, indices]
    x = indices % W + joff[0, indices]
    coords = torch.stack((x, y), dim=-1)
    coords = coords[scores > threshold]
    if return_indices:
        return coords, scores, indices
    return coords, scores


def bilinear_interpolate(feats: Tensor, points: Tensor) -> Tensor:
    """
    Applies bilinear interpolation to extract feature values at non-integer pixel
    locations specified by the points.

    Parameters
    ----------
    feats (Tensor): A 3D tensor representing feature maps. It should have
                    dimensions in the form of (C, H, W), where C is the number
                    of channels, H is the height, and W is the width.
    points (Tensor): A 2D tensor containing points at which to sample the features.
                    Each point is represented by its (x, y) coordinates. It should
                    have dimensions in the form of (N, 2), where N is the number of
                    points.

    Returns
    -------
    A tensor of shape (C, N) containing the interpolated values from the
        feature maps at each specified point.
    """
    assert feats.ndim == 3
    assert points.ndim == 2 and points.size(-1) == 2

    _, H, W = feats.shape
    x, y = torch.unbind(points, dim=1)
    # Calculate the bottom-left (x0, y0) and top-right (x1, y1) corners of the cells
    # containing the points. Clamp the values to ensure they are within the feature map.
    x0 = x.floor().clamp(0, W - 1)
    y0 = y.floor().clamp(0, H - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y1 = (y0 + 1).clamp(0, H - 1)
    # Convert the corner points to long type for use as indices in the tensor
    x0i, y0i, x1i, y1i = x0.long(), y0.long(), x1.long(), y1.long()

    # For each point, calculate a weighted average of the features at the four corners
    # of the cell containing the point. The weights depend on the proximity of the point
    # to each corner.
    bilerp_feats = (
        feats[:, y0i, x0i] * (y1 - y) * (x1 - x) +
        feats[:, y1i, x0i] * (y - y0) * (x1 - x) +
        feats[:, y0i, x1i] * (y1 - y) * (x - x0) +
        feats[:, y1i, x1i] * (y - y0) * (x - x0)
    )

    return bilerp_feats


def loi_sampling(feats: Tensor, lines: Tensor, steps: Tensor) -> Tensor:
    """
    Performs Line of Interest (LOI) sampling on the given feature map using specified lines
        and (bilinear) interpolation steps.

    Parameters
    ----------
    feats (Tensor): A 3D tensor of shape [C, H, W].
    lines (Tensor): A 2D tensor of shape [N, 4], where each row represents a line defined
                    by two points (x1, y1, x2, y2) in the feature space.
    steps (Tensor): A 1D tensor of shape [M] that defines the steps at which to interpolate
                    along each line.

    Returns
    -------
    Tensor: A tensor of shape [C, N, M]. This tensor contains the interpolated features along
            the given lines.
    """
    assert feats.ndim == 3
    assert lines.ndim == 2 and lines.size(1) == 4
    steps = steps.reshape(1, -1, 1)

    C = len(feats)
    N = len(lines)
    p1, p2 = lines[:, :2], lines[:, 2:]
    sampled_points = p1[:, None] * steps + p2[:, None] * (1 - steps) - 0.5
    sampled_points = sampled_points.flatten(end_dim=-2)
    interp_feats = bilinear_interpolate(feats, sampled_points)
    interp_feats = interp_feats.reshape(C, N, -1)

    return interp_feats


class MultitaskHead(nn.Module):
    def __init__(self, in_channels: int, task_sizes: Sequence[int]) -> None:
        super().__init__()

        inter_channels = in_channels // 4
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            ) for out_channels in task_sizes
        ])

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([head(x) for head in self.heads], dim=1)


def build_backbone(task_sizes: Sequence[int]):
    return StackedHourglass(
        Bottleneck,
        partial(MultitaskHead, task_sizes=task_sizes),
        num_classes=sum(task_sizes),
    )


class LCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer("lambda_", torch.linspace(0, 1, 32)[:, None])
        self.backbone = build_backbone([2, 1, 2])
        self.fc1 = nn.Conv2d(256, 128, kernel_size=1)
        self.fc2 = nn.Sequential(
            nn.Linear(1024 + 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        assert not self.training and len(images) == 1

        (features,), heatmaps_list = self.backbone(images)
        heatmaps = heatmaps_list[0]
        jloc, = heatmaps[:, 0:2].softmax(dim=1)[:, [1]]
        joff, = heatmaps[:, 3:5].sigmoid()[:, [1, 0]]   # assure [dx, dy] format

        junc_coords, _ = propose_junctions(jloc, joff, threshold=0.008)
        line_coords = self.propose_lines(junc_coords)

        loi_feats = self.fc1(features)
        line_feats = loi_sampling(loi_feats, line_coords, steps=self.lambda_)
        line_feats = line_feats.transpose_(0, 1)
        line_feats = F.max_pool1d(line_feats, kernel_size=4)
        line_feats = line_feats.flatten(start_dim=1)
        # concatenate dummy features
        line_feats = torch.cat((line_feats,
            torch.zeros(len(line_feats), 8, device=line_feats.device)), dim=-1)
        line_logits = self.fc2(line_feats).squeeze_(dim=-1)
        line_scores = line_logits.sigmoid()

        return line_coords, line_scores

    def propose_lines(self, junc_coords: Tensor) -> Tensor:
        device = junc_coords.device
        j1, j2 = torch.meshgrid(
            torch.arange(len(junc_coords), device=device),
            torch.arange(len(junc_coords), device=device),
            indexing="ij",
        )
        j1, j2 = j1.flatten(), j2.flatten()
        keep = j1 < j2
        j1, j2 = j1[keep], j2[keep]
        junc_indices = torch.stack((j1, j2), dim=-1)
        line_coords = junc_coords[junc_indices].reshape(-1, 4)
        return line_coords
