"""PyTorch Dataset with horizontal-flip augmentation and geometric features.

Each sample yields:
- image: normalized tensor of shape (3, H, W)
- side: long tensor, 0 = off-screen left, 1 = off-screen right
- regression targets: tensor of shape (4,) - dist_from_edge, width, height, y_center
  (all z-score normalized)
- direction: long tensor, 0 = walking out of frame, 1 = walking into frame
- geometric features: tensor of shape (19,)

Horizontal flip augmentation:
- The image is mirrored
- side label is swapped (0 <-> 1)
- direction label is swapped (encodes direction relative to off-screen position)
- distance_from_edge, width, height, y_center are unchanged (symmetric)
- geometric features are mirrored via flip_geometric_features()
"""

from __future__ import annotations

import random
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .data import normalize_target
from .features import flip_geometric_features


class ShadowDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        target_stats: dict[str, dict[str, float]],
        input_size: tuple[int, int],
        channel_means: list[float],
        channel_stds: list[float],
        augment: bool = False,
        flip_prob: float = 0.5,
    ):
        self.samples = samples
        self.ts = target_stats
        self.augment = augment
        self.flip_prob = flip_prob

        self.base_tf = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(channel_means, channel_stds),
            ]
        )
        self.aug_tf = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(channel_means, channel_stds),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        side = s["side"]
        direction = s["direction"]
        geo = s["geo"].copy()

        do_flip = self.augment and random.random() < self.flip_prob
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            side = 1 - side
            # We also flip the direction label - in this dataset, direction encodes
            # "walking into frame" relative to which side of the screen the
            # person is on, so mirroring the image swaps it.
            direction = 1 - direction
            geo = flip_geometric_features(geo)

        img_t = self.aug_tf(img) if self.augment else self.base_tf(img)

        reg = torch.tensor(
            [
                normalize_target(s["distance_from_edge"], "distance_from_edge", self.ts),
                normalize_target(s["bbox_width"], "bbox_width", self.ts),
                normalize_target(s["bbox_height"], "bbox_height", self.ts),
                normalize_target(s["y_center"], "y_center", self.ts),
            ],
            dtype=torch.float32,
        )

        return (
            img_t,
            torch.tensor(side, dtype=torch.long),
            reg,
            torch.tensor(direction, dtype=torch.long),
            torch.from_numpy(geo).float(),
        )
