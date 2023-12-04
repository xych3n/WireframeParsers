from typing import Literal

import torch
import torchvision.models
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d

from .backbone.detr import Transformer


class LETR(nn.Module):
    def __init__(self, backbone: Literal["resnet50", "resnet101"]):
        super().__init__()

        self.backbone = IntermediateLayerGetter(
            getattr(torchvision.models, backbone)(norm_layer=FrozenBatchNorm2d),
            {"layer3": "layer3", "layer4": "layer4"},
        )
        self.input_proj1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.input_proj2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.query_embed = nn.Embedding(num_embeddings=1000, embedding_dim=256)
        self.transformer1 = Transformer(d_model=256)
        self.transformer2 = Transformer(d_model=256)
        self.class_embed = nn.Linear(256, 2)
        self.lines_embed = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

    def forward(self, images: Tensor):
        assert not self.training and len(images) == 1

        line_entities = self.query_embed.weight[:, None]
        features_dict = self.backbone(images)

        # Stage 1
        features = self.input_proj1(features_dict["layer4"])
        line_entities = self.transform(features, line_entities, self.transformer1)
        # Stage 2
        features = self.input_proj2(features_dict["layer3"])
        line_entities = self.transform(features, line_entities, self.transformer2)

        line_entities = line_entities.squeeze_()
        line_coords = self.lines_embed(line_entities).sigmoid() * 128
        line_logits = self.class_embed(line_entities)
        line_scores = line_logits.softmax(dim=-1)[:, 0]

        return line_coords, line_scores

    def transform(self, features: Tensor, line_entities: Tensor, transformer):
        pos_embed = self.positional_encoding(features, num_feats=128)
        to_sequence = lambda t: t.flatten(start_dim=-2).permute(2, 0, 1)
        features = to_sequence(features)
        tgt = torch.zeros_like(line_entities)
        pos_embed = to_sequence(pos_embed)
        return transformer(features, tgt, pos_embed=pos_embed, query_embed=line_entities)

    @staticmethod
    def positional_encoding(src: Tensor, num_feats: int, temperature: int = 10000):
        B, _, H, W = src.shape
        assert B == 1
        mask = torch.zeros((1, H, W), dtype=bool, device=src.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * 2 * torch.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * 2 * torch.pi
        dim_t = torch.arange(num_feats, dtype=torch.float32, device=mask.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
