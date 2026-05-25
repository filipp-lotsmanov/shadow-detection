"""Shadow detection model.

Architecture:
- ResNet-50 backbone (ImageNet-pretrained) produces 2048-d image features
- 19 hand-crafted geometric features are concatenated -> 2067-d
- Shared projection (Linear + BatchNorm + ReLU + Dropout) -> 512-d
- Three task-specific heads:
    1. Side classification (left vs right)
    2. Bbox regression (4 decomposed targets: dist_from_edge, width, height, y_center)
    3. Direction classification (toward vs away from frame)
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

NUM_GEO_FEATURES = 19


class ShadowModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
        num_geo_features: int = NUM_GEO_FEATURES,
    ):
        super().__init__()

        if backbone != "resnet50":
            raise NotImplementedError(f"Backbone {backbone!r} not supported yet")

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)
        # Strip the final classifier; keep avgpool
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.num_geo_features = num_geo_features
        feat_dim = 2048 + num_geo_features

        self.shared_proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.side_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(
        self, x: torch.Tensor, geo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x).flatten(1)
        features = torch.cat([features, geo], dim=1)
        shared = self.shared_proj(features)
        return self.side_head(shared), self.regression_head(shared), self.direction_head(shared)
