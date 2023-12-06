from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .lcnn import build_backbone, bilinear_interpolate, loi_sampling, propose_junctions


def deocde_hafm(angs: Tensor, dist: Tensor, bias: Optional[Tensor] = None, bias_span: int = 1, scale: float = 5.0) -> Tensor:
    """
    Decodes a Holistically-Attracted Field Map (HAFM) into line coordinates.

    Parameters
    ----------
    angs : Tensor of shape [B, 3, H, W]
    dist : Tensor of shape [B, 1, H, W]
    bias : None or Tensor of shape [B, 1, H, W]
        a bias value to be applied to the dist.
    bias_span : int
        the range of values to be generated for bias adjustment. Default value is 1.
    scale : float
        a scale factor to adjust the dist. Default value is 5.0.

    Returns
    -------
    Tensor of shape [B, C, H, W, 4]
        A tensor containing coordinates for the start and end points of lines.  C is 3 if a bias is applied, otherwise 1.
    """
    assert angs.ndim == 4 and angs.size(1) == 3
    assert dist.ndim == 4 and dist.size(1) == 1
    _, _, H, W = angs.shape
    device = angs.device

    if bias is not None:
        assert bias.ndim == 4 and bias.size(1) == 1
        bias_range = torch.arange(-bias_span, bias_span + 1,
            dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
        dist = (dist + bias * bias_range).clamp(0, 1)

    angle0 = (angs[:, 0:1] - 0.5) * torch.pi * 2
    angle1 = angs[:, 1:2] * torch.pi / 2
    angle2 = -angs[:, 2:3] * torch.pi / 2

    cos_angle0 = angle0.cos()
    sin_angle0 = angle0.sin()
    tan_angle1 = angle1.tan()
    tan_angle2 = angle2.tan()

    x1 = (cos_angle0 - sin_angle0 * tan_angle1) * dist * scale
    y1 = (sin_angle0 + cos_angle0 * tan_angle1) * dist * scale
    x2 = (cos_angle0 - sin_angle0 * tan_angle2) * dist * scale
    y2 = (sin_angle0 + cos_angle0 * tan_angle2) * dist * scale

    y0, x0 = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    y0, x0 = y0[None, None], x0[None, None]

    x1 = (x1 + x0).clamp(0, W - 1)
    y1 = (y1 + y0).clamp(0, H - 1)
    x2 = (x2 + x0).clamp(0, W - 1)
    y2 = (y2 + y0).clamp(0, H - 1)

    lines = torch.stack((x1, y1, x2, y2), dim=-1)

    return lines


class HAWPv1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer("tspan", torch.linspace(0, 1, 32)[None, None])

        self.backbone = build_backbone([3, 1, 1, 2, 2])
        self.fc1 = nn.Conv2d(256, 128, kernel_size=1)
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        assert not self.training and len(images) == 1

        (features,), heatmaps_list = self.backbone(images)
        heatmaps = heatmaps_list[0]
        angs = heatmaps[:, 0:3].sigmoid()
        dist = heatmaps[:, 3:4].sigmoid()
        bias = heatmaps[:, 4:5].sigmoid()
        jloc, = heatmaps[:, 5:7].softmax(dim=1)[:, [1]]
        joff, = heatmaps[:, 7:9].sigmoid()

        junc_coords, _ = propose_junctions(jloc, joff, threshold=0.008)
        line_coords = deocde_hafm(angs, dist, bias).reshape(-1, 4)
        line_coords = self.propose_lines(junc_coords, line_coords)

        loi_feats = self.fc1(features)
        line_feats = loi_sampling(loi_feats, line_coords, steps=self.tspan)
        line_feats = line_feats.transpose_(0, 1)
        line_feats = F.max_pool1d(line_feats, kernel_size=4)
        line_feats = line_feats.flatten(start_dim=1)
        line_logits = self.fc2(line_feats).squeeze_(dim=-1)
        line_scores = line_logits.sigmoid()

        return line_coords, line_scores

    def propose_lines(self, junc_coords: Tensor, line_coords: Tensor) -> Tensor:
        _, j1 = ((junc_coords[:, None] - line_coords[:, :2]) ** 2).sum(dim=-1).min(dim=0)
        _, j2 = ((junc_coords[:, None] - line_coords[:, 2:]) ** 2).sum(dim=-1).min(dim=0)
        j1, j2 = torch.min(j1, j2), torch.max(j1, j2)
        keep = j1 < j2
        j1, j2 = j1[keep], j2[keep]
        junction_pairs = torch.stack((j1, j2), dim=-1)
        junction_pairs = junction_pairs.unique(dim=0)
        line_coords = junc_coords[junction_pairs].reshape(-1, 4)
        return line_coords


class HAWPv2(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("tspan", torch.linspace(0, 1, 32)[None, None])

        self.backbone = build_backbone([3, 1, 1, 2, 2])
        self.fc1 = nn.Conv2d(256, 128, kernel_size=1)
        self.fc3 = nn.Conv2d(256, 4, kernel_size=1)
        self.fc4 = nn.Conv2d(256, 4, kernel_size=1)
        self.fc2 = nn.Sequential(
            nn.Linear(128 * 2 + (32 - 2) * 4 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )
        self.fc2_res = nn.Sequential(
            nn.Linear(2 * (32 - 2) * 4, 1024),
            nn.ReLU(inplace=True),
        )
        self.fc2_head = nn.Linear(1024, 2)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        assert not self.training and len(images) == 1

        (features,), heatmaps_list = self.backbone(images)
        heatmaps = heatmaps_list[0]
        angs = heatmaps[:, 0:3].sigmoid()
        dist = heatmaps[:, 3:4].sigmoid()
        bias = heatmaps[:, 4:5].sigmoid()
        jloc, = heatmaps[:, 5:7].softmax(dim=1)[:, [1]]
        joff, = heatmaps[:, 7:9].sigmoid()

        junc_coords, _ = propose_junctions(jloc, joff, k=300, threshold=0.008)
        line_coords = deocde_hafm(angs, dist, bias, bias_span=2).reshape(-1, 4)
        line_coords, lines_raw = self.propose_lines(junc_coords, line_coords)

        loi_features = self.fc1(features)
        e1_feats = bilinear_interpolate(loi_features, line_coords[:, :2] - .5)
        e2_feats = bilinear_interpolate(loi_features, line_coords[:, 2:] - .5)
        loi_features_thin = self.fc3(features)
        loi_features_aux = self.fc4(features)
        thin_feats = loi_sampling(loi_features_thin, line_coords, steps=self.tspan[..., 1:-1])
        aux_feats = loi_sampling(loi_features_aux, lines_raw, steps=self.tspan[..., 1:-1])

        e1_feats = e1_feats.transpose_(0, 1)
        e2_feats = e2_feats.transpose_(0, 1)
        thin_feats = thin_feats.transpose_(0, 1).flatten(start_dim=1)
        aux_feats = aux_feats.transpose_(0, 1).flatten(start_dim=1)
        line_feats = torch.cat((e1_feats, e2_feats, thin_feats, aux_feats), dim=-1)
        line_logits = self.fc2_head(self.fc2(line_feats) + self.fc2_res(torch.cat((thin_feats, aux_feats), dim=-1)))
        line_scores = line_logits.softmax(dim=-1)[:, 1]

        return line_coords, line_scores

    def propose_lines(self, junc_coords: Tensor, line_coords: Tensor, threshold: float = 10.0) -> Tensor:
        dist1, j1 = ((junc_coords[:, None] - line_coords[:, :2]) ** 2).sum(dim=-1).min(dim=0)
        dist2, j2 = ((junc_coords[:, None] - line_coords[:, 2:]) ** 2).sum(dim=-1).min(dim=0)
        j1, j2 = torch.min(j1, j2), torch.max(j1, j2)
        keep = (j1 < j2) & (dist1 < threshold) & (dist2 < threshold)
        j1, j2 = j1[keep], j2[keep]
        junction_pairs = torch.stack((j1, j2), dim=-1)

        junction_pairs, order_inv = junction_pairs.unique(dim=0, return_inverse=True)
        order = torch.arange(len(order_inv), dtype=order_inv.dtype, device=order_inv.device)
        order, order_inv = order.flip(dims=[0]), order_inv.flip(dims=[0])
        order_scatter = order_inv.new_empty(junction_pairs.size(0)).scatter_(0, order_inv, order)
        lines_init = line_coords[keep][order_scatter]

        line_coords = junc_coords[junction_pairs].reshape(-1, 4)

        return line_coords, lines_init
