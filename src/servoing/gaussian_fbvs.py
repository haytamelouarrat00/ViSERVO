from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from src.camera import create_camera_from_K
from src.control import geometric_error, interaction_matrix, normalize_features, velocity
from src.features import XFeatMatcher
from src.render_pipeline import render_gaussian_view
from src.servoing.target_io import _load_and_fit_target
from src.servoing.visualization import (
    _compose_current_target_view,
    _draw_iteration_features,
    _make_frame_output_dir,
    _make_video_writer,
    _safe_imshow,
)


def _init_scene_and_camera(scene_path: str | Path, K: np.ndarray, device: torch.device):
    """Load Gaussian scene and camera intrinsics."""
    from src.gs import GaussianSplattingScene

    scene_obj = GaussianSplattingScene(device=str(device))
    scene_obj.load_from_ply(str(scene_path))

    camera = create_camera_from_K(K, width=1920, height=1080)

    cam_h = int(camera.intrinsics.height)
    cam_w = int(camera.intrinsics.width)
    return scene_obj, camera, cam_w, cam_h


def _init_features_and_world_points(
    *,
    rendered_gray: np.ndarray,
    rendered_depth: np.ndarray,
    real_gray: np.ndarray,
    camera,
    camera_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize tracked/desired features from first frame and back-project to world.

    Returns:
        tracked_features_px (N,2)
        desired_features_px (N,2)
        desired_features_norm (N,2)
        feature_points_world (N,3)
    """
    xfeat_matcher = XFeatMatcher(top_k=4096)
    match_result = xfeat_matcher.match(rendered_gray, real_gray)
    kp1, kp2, matches = (
        match_result.keypoints_ref,
        match_result.keypoints_qry,
        match_result.matches,
    )

    if not matches:
        raise RuntimeError("[fbvs] No feature matches found during initialization.")

    tracked_features_px = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    desired_features_px = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    desired_features_norm = normalize_features(desired_features_px, camera=camera)

    # sample depth at tracked pixels
    h, w = rendered_depth.shape[:2]
    uv = np.round(tracked_features_px).astype(int)
    uv[:, 0] = np.clip(uv[:, 0], 0, w - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, h - 1)
    init_depths = rendered_depth[uv[:, 1], uv[:, 0]]
    init_depths = np.clip(init_depths, 1e-6, None).astype(np.float32)

    camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
    feature_points_world = camera.back_project_to_world(
        tracked_features_px, init_depths
    ).astype(np.float32)

    return (
        tracked_features_px,
        desired_features_px,
        desired_features_norm,
        feature_points_world,
    )


def fbvs(
    scene: str | Path,
    initial_pose: Sequence[float],
    desired_view: str | Path,
    *,
    max_iters: int = 1000,
    error_tolerance: float = 0.12,
    gain: float = 0.5,
    dt: float = 1.0,
    vis_window_name: str = "FBVS Current vs Target",
    vis_wait_ms: int = 1,
    K: np.ndarray = None,
    save_video: bool = True,
    video_path: str = "output.mp4",
    fps: float = 10.0,
    save_frames: bool = True,
    frames_dir: str | Path = "debug_frames",
):
    pass


__all__ = ["fbvs", "_init_scene_and_camera", "_init_features_and_world_points"]
