from __future__ import annotations

import numpy as np

from pvs_framework.camera import CameraIntrinsics
from pvs_framework.pvs_math import compute_interaction_matrix, zero_mean_normalize


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    u0 = np.clip(u0, 0, w - 1)
    v0 = np.clip(v0, 0, h - 1)

    du = u - u0
    dv = v - v0

    I00 = image[v0, u0]
    I10 = image[v0, u1]
    I01 = image[v1, u0]
    I11 = image[v1, u1]

    return (
        (1.0 - du) * (1.0 - dv) * I00
        + du * (1.0 - dv) * I10
        + (1.0 - du) * dv * I01
        + du * dv * I11
    )


def _intrinsics(width: int, height: int) -> CameraIntrinsics:
    K = np.array(
        [[90.0, 0.0, (width - 1) * 0.5], [0.0, 95.0, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraIntrinsics.from_K(width=width, height=height, K=K)


def test_interaction_matrix_shape() -> None:
    h, w = 12, 16
    image = np.random.default_rng(0).normal(size=(h, w)).astype(np.float64)
    depth = np.full((h, w), 2.0, dtype=np.float64)
    intr = _intrinsics(w, h)

    L = compute_interaction_matrix(zero_mean_normalize(image), depth, intr)
    assert L.shape == (h * w, 6)


def test_interaction_matrix_matches_finite_difference() -> None:
    h, w = 24, 28
    yy, xx = np.indices((h, w), dtype=np.float64)
    image = np.sin(0.17 * xx) + 0.5 * np.cos(0.13 * yy) + 0.2 * np.sin(0.07 * (xx + yy))
    image = zero_mean_normalize(image)

    depth = np.full((h, w), 2.0, dtype=np.float64)
    intr = _intrinsics(w, h)
    L = compute_interaction_matrix(image, depth, intr)

    delta = np.array([1e-6, -2e-6, 1.5e-6, 1e-6, -1.2e-6, 0.8e-6], dtype=np.float64)
    predicted = (L @ delta).reshape(h, w)

    fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - cx) / fx
    y = (v - cy) / fy
    Z = depth

    vx, vy, vz, wx, wy, wz = delta
    xdot = (-vx / Z) + (x * vz / Z) + (x * y * wx) - ((1.0 + x * x) * wy) + (y * wz)
    ydot = (-vy / Z) + (y * vz / Z) + ((1.0 + y * y) * wx) - (x * y * wy) - (x * wz)

    x2 = x + xdot
    y2 = y + ydot
    u2 = fx * x2 + cx
    v2 = fy * y2 + cy

    numeric = _bilinear_sample(image, u2, v2) - image

    mask = (u2 >= 1.0) & (u2 <= w - 2.0) & (v2 >= 1.0) & (v2 <= h - 2.0)
    mae = float(np.mean(np.abs(predicted[mask] - numeric[mask])))
    assert mae < 2e-3
