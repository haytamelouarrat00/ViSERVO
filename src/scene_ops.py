from __future__ import annotations

from typing import Tuple

import numpy as np

from src.gs import GaussianParameters
from src.rotations import euler_to_quaternion, quat_to_matrix


def scene_bounds(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute min/max/center/radius for Nx3 points."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.linalg.norm(maxs - mins)) * 0.5, 0.2)
    return mins, maxs, center, radius


def quat_multiply_left(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Compute q_left * q_right for q_right of shape (N, 4), format [w, x, y, z]."""
    w1, x1, y1, z1 = np.asarray(q_left, dtype=np.float32).reshape(4)
    q_right = np.asarray(q_right, dtype=np.float32)
    if q_right.ndim != 2 or q_right.shape[1] != 4:
        raise ValueError(f"q_right must have shape (N,4), got {q_right.shape}")

    w2, x2, y2, z2 = q_right[:, 0], q_right[:, 1], q_right[:, 2], q_right[:, 3]
    out = np.column_stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    ).astype(np.float32)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return out / norm


def apply_scene_pose(
    params: GaussianParameters, pose: np.ndarray
) -> GaussianParameters:
    """
    Apply a 6-DoF world pose [x, y, z, rx, ry, rz] to all gaussians.
    Rotation convention is [pitch, roll, yaw].
    """
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (6,):
        raise ValueError(f"scene pose must have shape (6,), got {pose.shape}")

    t = pose[:3]
    euler = pose[3:]
    q_scene = euler_to_quaternion(euler).astype(np.float32)
    r_scene = quat_to_matrix(q_scene).astype(np.float32)

    means_new = (r_scene @ params.means.T).T + t[None, :]
    rotations_new = quat_multiply_left(q_scene, params.rotations.astype(np.float32))

    return GaussianParameters(
        means=means_new.astype(np.float32),
        scales=params.scales.astype(np.float32),
        rotations=rotations_new.astype(np.float32),
        opacities=params.opacities.astype(np.float32),
        colors=params.colors.astype(np.float32),
    )


def add_world_frame_gaussians(
    params: GaussianParameters,
    axis_length: float,
    samples_per_axis: int,
    axis_scale: float,
    axis_opacity: float = 1.0,
) -> GaussianParameters:
    """Append RGB axis gaussians at the world origin."""
    if params.colors.ndim != 2 or params.colors.shape[1] != 3:
        raise ValueError(
            "World-frame overlay currently supports RGB colors only (N, 3)."
        )

    t = np.linspace(0.0, axis_length, samples_per_axis, dtype=np.float32)
    x_axis = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    y_axis = np.stack([np.zeros_like(t), t, np.zeros_like(t)], axis=1)
    z_axis = np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)
    frame_means = np.vstack([x_axis, y_axis, z_axis])

    frame_colors = np.vstack(
        [
            np.tile([1.0, 0.0, 0.0], (samples_per_axis, 1)),
            np.tile([0.0, 1.0, 0.0], (samples_per_axis, 1)),
            np.tile([0.0, 0.0, 1.0], (samples_per_axis, 1)),
        ]
    ).astype(np.float32)

    frame_scales = np.full((frame_means.shape[0], 3), axis_scale, dtype=np.float32)
    frame_rotations = np.tile([1.0, 0.0, 0.0, 0.0], (frame_means.shape[0], 1)).astype(
        np.float32
    )
    frame_opacities = np.full((frame_means.shape[0], 1), axis_opacity, dtype=np.float32)

    return GaussianParameters(
        means=np.vstack([params.means, frame_means]).astype(np.float32),
        scales=np.vstack([params.scales, frame_scales]).astype(np.float32),
        rotations=np.vstack([params.rotations, frame_rotations]).astype(np.float32),
        opacities=np.vstack([params.opacities, frame_opacities]).astype(np.float32),
        colors=np.vstack([params.colors, frame_colors]).astype(np.float32),
    )
