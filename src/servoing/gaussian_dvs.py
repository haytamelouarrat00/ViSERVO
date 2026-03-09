from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from src.camera import create_camera_from_K
from src.control import (
    geometric_error,
    interaction_matrix,
    normalize_features,
    velocity, photometric_error,
    normalize_image
)
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
import cv2

def _infer_camera_size_from_K(K: np.ndarray) -> tuple[int, int]:
    """Infer image width/height from principal point when possible."""
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K_arr.shape}")

    width = int(round(float(K_arr[0, 2]) * 2.0))
    height = int(round(float(K_arr[1, 2]) * 2.0))
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid inferred camera size from K: width={width}, height={height}"
        )
    return width, height


def _init_scene_and_camera(scene_path: str | Path, K: np.ndarray, device: torch.device):
    """Load Gaussian scene and camera intrinsics."""
    from src.gs import GaussianSplattingScene

    scene_obj = GaussianSplattingScene(device=str(device))
    scene_obj.load_from_ply(str(scene_path))

    if K is None:
        # Default intrinsics used by current playroom setup.
        K = np.array(
            [
                [1040.0073037593279, 0.0, 632.0],
                [0.0, 1040.1927566661841, 416.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:
        K = np.asarray(K, dtype=np.float32)

    cam_w, cam_h = _infer_camera_size_from_K(K)
    camera = create_camera_from_K(K=K, width=cam_w, height=cam_h)

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

    tracked_features_px = np.array(
        [kp1[m.queryIdx].pt for m in matches], dtype=np.float32
    )
    desired_features_px = np.array(
        [kp2[m.trainIdx].pt for m in matches], dtype=np.float32
    )

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


def get_pose_colmap(
    image_id: int, path: str | Path = "data/playroom/info/images.txt"
) -> Optional[np.ndarray]:
    """
    Read a COLMAP image pose and return camera pose in world frame.

    COLMAP stores image poses as world-to-camera (w2c):
        x_cam = R_w2c * x_world + t_w2c
    This function converts them to camera-to-world (c2w) pose:
        [x, y, z, rx, ry, rz] where xyz is camera center in world and
        r* are XYZ Euler angles of R_c2w.
    """
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if int(parts[0]) != int(image_id):
            continue

        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])

        # COLMAP: world -> camera
        R_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tx, ty, tz], dtype=np.float64)

        # Convert to camera -> world
        R_c2w = R_w2c.T
        C_world = -R_c2w @ t_w2c
        euler_c2w = R.from_matrix(R_c2w).as_euler("xyz")

        return np.array(
            [
                C_world[0],
                C_world[1],
                C_world[2],
                euler_c2w[0],
                euler_c2w[1],
                euler_c2w[2],
            ],
            dtype=np.float32,
        )

    return None


def gaussian_dvs(
    scene: str | Path,
    initial_pose: Sequence[float],
    desired_view: str | Path,
    *,
    max_iters: int = 20,
    cost_tolerance: float = 1e3,
    error_tolerance: Optional[float] = None,
    gain: float = 4.0,
    use_zero_mean_normalization: bool = True,
    use_gradient_magnitude: bool = True,
    K: np.ndarray = None,
    dt: float = 1.0,
    vis_window_name: str = "DVS Gaussian Current vs Target",
    vis_wait_ms: int = 1,
    save_video: bool = False,
    video_path: str = "output_gs_dvs.mp4",
    fps: float = 10.0,
    save_frames: bool = False,
    frames_dir: str | Path = "debug_frames_gs_dvs",
    verbose: bool = False,
    desired_pose: np.ndarray = None,
):
    """
    Direct Visual Servoing (DVS) loop with Gaussian Splatting scene and ZN Gauss-Newton control.
    """
    camera_pose = np.asarray(initial_pose, dtype=np.float32).copy()
    if camera_pose.size != 6:
        raise ValueError(
            f"initial_pose must have 6 elements, got shape {camera_pose.shape}"
        )
    scene_pose_np = np.zeros(6, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_obj, camera, cam_w, cam_h = _init_scene_and_camera(scene, K, device)
    real_bgr, real_gray = _load_and_fit_target(desired_view, cam_w, cam_h)
    real_gray = normalize_image(real_gray)
    video_writer = None
    enable_video = bool(verbose and save_video)
    if enable_video:
        video_writer, _ = _make_video_writer(video_path, cam_w, cam_h, fps=fps)

    frame_output_dir = None
    saved_frame_count = 0
    enable_frames = bool(verbose and save_frames)
    if enable_frames:
        frame_output_dir = _make_frame_output_dir(frames_dir)
        print(f"[fbvs_gaussian_redetect] Saving debug frames to {frame_output_dir}")

    converged = False
    iteration = 0
    camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
    result = render_gaussian_view(
        scene=scene_obj,
        camera_pose=camera_pose,
        scene_pose=scene_pose_np,
        device=str(device),
        camera=camera,
        return_depth=True,
    )
    render_rgb = np.asarray(result.rgb, dtype=np.float32)

    render_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    render_gray = normalize_image(render_gray)
    render_depth = np.asarray(result.depth, dtype=np.float32)

    ph_diff = render_gray - real_gray
    if verbose:
        print(ph_diff.min(), ph_diff.max(), ph_diff.mean(), ph_diff.std())

        err_vis = np.clip((ph_diff - ph_diff.min()) / (ph_diff.max() - ph_diff.min() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
        err_color = cv2.applyColorMap(err_vis, cv2.COLORMAP_JET)

        def to_u8(img):
            return np.clip((img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0, 0, 255).astype(np.uint8)

        render_u8 = to_u8(render_gray)
        real_u8 = to_u8(real_gray)
        render_bgr = cv2.cvtColor(render_u8, cv2.COLOR_GRAY2BGR)
        real_bgr = cv2.cvtColor(real_u8, cv2.COLOR_GRAY2BGR)
        panel = np.hstack([render_bgr, real_bgr, err_color])
        cv2.imwrite("debug_ph_diff.png", panel)

    error, cost = photometric_error(render_gray, real_gray)
    if verbose:
        print('[Gaussian DVS] Cost: ', cost)


    