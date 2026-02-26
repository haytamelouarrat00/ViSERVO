from __future__ import annotations

import cv2
import numpy as np

from pvs_framework.camera import CameraIntrinsics, exp_se3
from pvs_framework.controller import PVSController


class ShiftRenderer:
    def __init__(
        self,
        base_image: np.ndarray,
        depth_map: np.ndarray,
        pixels_per_meter: float = 300.0,
    ):
        self._base = base_image.astype(np.float64)
        self._depth = depth_map.astype(np.float64)
        self._ppm = float(pixels_per_meter)
        self._h, self._w = self._base.shape

    def _render_from_pose(self, pose: np.ndarray) -> np.ndarray:
        tx = float(pose[0, 3])
        ty = float(pose[1, 3])
        M = np.array(
            [[1.0, 0.0, -tx * self._ppm], [0.0, 1.0, -ty * self._ppm]],
            dtype=np.float64,
        )
        return cv2.warpAffine(
            self._base,
            M,
            (self._w, self._h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def render_color(self, pose: np.ndarray) -> np.ndarray:
        return self._render_from_pose(pose)

    def render_depth(self, pose: np.ndarray) -> np.ndarray:
        return self._depth.copy()

    def render_color_and_depth(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.render_color(pose), self.render_depth(pose)


def _intrinsics(width: int, height: int) -> CameraIntrinsics:
    K = np.array(
        [[140.0, 0.0, (width - 1) * 0.5], [0.0, 140.0, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraIntrinsics.from_K(width=width, height=height, K=K)


def test_controller_single_step_moves_closer() -> None:
    h, w = 48, 64
    yy, xx = np.indices((h, w), dtype=np.float64)
    base = np.sin(0.2 * xx) + np.cos(0.15 * yy) + 0.1 * np.sin(0.1 * (xx + yy))
    depth = np.full((h, w), 2.0, dtype=np.float64)

    renderer = ShiftRenderer(base, depth)
    intr = _intrinsics(w, h)
    controller = PVSController(
        renderer=renderer,
        intrinsics=intr,
        lambda_gain=0.1,
        convergence_threshold=1e-10,
        max_iterations=50,
    )

    desired_pose = np.eye(4, dtype=np.float64)
    initial_pose = np.eye(4, dtype=np.float64)
    initial_pose[0, 3] = 0.005  # 5mm offset

    controller.set_desired_image(desired_pose)

    twist, cost_before = controller.step(initial_pose)
    next_pose = initial_pose @ exp_se3(twist)
    _, cost_after = controller.step(next_pose)

    dist_before = float(np.linalg.norm(initial_pose[:3, 3] - desired_pose[:3, 3]))
    dist_after = float(np.linalg.norm(next_pose[:3, 3] - desired_pose[:3, 3]))

    assert cost_after <= cost_before
    assert dist_after <= dist_before
