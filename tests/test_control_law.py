from __future__ import annotations

import numpy as np

from pvs_framework.camera import CameraIntrinsics
from pvs_framework.controller import PVSController


class DummyRenderer:
    def __init__(self, image: np.ndarray, depth: np.ndarray):
        self._image = image.astype(np.float64)
        self._depth = depth.astype(np.float64)

    def render_color(self, pose: np.ndarray) -> np.ndarray:
        return self._image.copy()

    def render_depth(self, pose: np.ndarray) -> np.ndarray:
        return self._depth.copy()

    def render_color_and_depth(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.render_color(pose), self.render_depth(pose)


def _make_intrinsics(width: int, height: int) -> CameraIntrinsics:
    K = np.array(
        [[120.0, 0.0, (width - 1) * 0.5], [0.0, 120.0, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraIntrinsics.from_K(width=width, height=height, K=K)


def test_control_twist_is_zero_when_current_equals_desired() -> None:
    h, w = 20, 24
    yy, xx = np.indices((h, w), dtype=np.float64)
    image = np.sin(xx * 0.2) + np.cos(yy * 0.3)
    depth = np.full((h, w), 2.0, dtype=np.float64)

    renderer = DummyRenderer(image=image, depth=depth)
    intrinsics = _make_intrinsics(w, h)
    controller = PVSController(renderer=renderer, intrinsics=intrinsics, lambda_gain=0.5)

    pose = np.eye(4, dtype=np.float64)
    controller.set_desired_image(pose)
    twist, cost = controller.step(pose)

    assert cost <= 1e-10
    assert np.linalg.norm(twist) <= 1e-8
