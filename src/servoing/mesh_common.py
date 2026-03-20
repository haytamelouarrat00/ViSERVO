from __future__ import annotations

import cv2
import numpy as np

from src.camera import VirtualCamera
from src.mesh_scene import MeshScene
from src.servoing.visualization import render_rgb_to_u8


def render_view(camera: VirtualCamera, scene: MeshScene):
    rgb = scene.render_from_virtual_camera(camera)
    rgb_u8 = render_rgb_to_u8(rgb)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def get_features_depth(features: np.ndarray, depth_map: np.ndarray):
    coords = np.array(features).astype(int)
    x, y = coords[:, 0], coords[:, 1]
    h, w = depth_map.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return depth_map[y, x]


def default_mesh_intrinsics() -> np.ndarray:
    """Return the default intrinsics used by mesh FBVS/DVS loops."""
    return np.array(
        [
            [1158.3, 0.0, 649.0],
            [0.0, 1153.53, 483.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


__all__ = ["render_view", "get_features_depth", "default_mesh_intrinsics"]
