"""19 hand-crafted geometric features extracted from each input image.

Copied verbatim from the training package - must remain bit-identical so the
model sees the same feature representation it was trained on.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

NUM_FEATURES = 19


def extract_geometric_features(img_array: np.ndarray) -> np.ndarray:
    """Extract 19 hand-crafted geometric features from a raw RGB image."""
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    gray = np.mean(img_array.astype(np.float32), axis=2)
    H, W = gray.shape

    rm = np.median(gray, axis=1, keepdims=True)
    sm = gray < (rm * 0.85)

    rs = int(H * 0.4)
    road = sm[rs:, :]

    f = np.zeros(NUM_FEATURES, dtype=np.float32)

    f[0] = np.mean(gray[int(H * 0.625):, :20]) / 255
    f[1] = np.mean(gray[int(H * 0.625):, -20:]) / 255
    f[2] = f[0] / (f[1] + 1e-6)

    ys, xs = np.where(road)
    if len(xs) > 50:
        f[3] = np.mean(xs) / W
        f[4] = (np.mean(ys) + rs) / H
        f[5] = np.std(xs) / W
        f[6] = np.std(ys) / H
        f[7] = len(xs) / (road.shape[0] * road.shape[1])

        lm = np.sum(road[:, : W // 2])
        rm2 = np.sum(road[:, W // 2:])
        f[8] = lm / (lm + rm2 + 1e-6)

        cd = np.sum(road, axis=0).astype(float)
        f[9] = np.argmax(cd) / W
        f[10] = np.average(np.arange(W).astype(float), weights=cd + 1e-6) / W
        f[11] = np.sum(road[:, :30]) / (road.shape[0] * 30)
        f[12] = np.sum(road[:, -30:]) / (road.shape[0] * 30)

        if len(xs) > 100:
            cov = np.cov(xs - np.mean(xs), ys - np.mean(ys))
            _, ev = np.linalg.eigh(cov)
            f[13] = np.arctan2(ev[1, 1], ev[0, 1]) / np.pi
    else:
        f[3:14] = 0.5

    bl2 = gaussian_filter(gray, sigma=20)
    sd = bl2 - gray
    r2 = sd[int(H * 0.58):, :]
    m2 = r2 > 8
    y2, x2 = np.where(m2)
    if len(x2) > 50:
        f[14] = np.mean(r2[m2]) / 100
        f[15] = np.max(r2[m2]) / 100
        f[16] = np.mean(np.abs(sobel(r2, axis=1))[m2]) / 50
        f[17] = len(x2) / ((np.max(y2) - np.min(y2) + 1) * (np.max(x2) - np.min(x2) + 1) + 1e-6)
        f[18] = np.percentile(r2[m2], 90) / 100

    return f


def flip_geometric_features(geo: np.ndarray) -> np.ndarray:
    """Mirror the 19 features to match a horizontally flipped image.

    Returns a NEW array - does not mutate the input. Use after flipping the image
    so the geometric features are consistent with the flipped pixels.
    """
    g = geo.copy()
    # Swap left/right pairs
    g[0], g[1] = geo[1], geo[0]
    g[2] = g[0] / (g[1] + 1e-6)
    # Mirror x-coordinates around 0.5
    g[3] = 1 - geo[3]
    g[8] = 1 - geo[8]
    g[9] = 1 - geo[9]
    g[10] = 1 - geo[10]
    # Swap left/right edge density
    g[11], g[12] = geo[12], geo[11]
    return g
