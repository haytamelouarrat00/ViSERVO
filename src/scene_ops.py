from __future__ import annotations

from typing import Tuple, Sequence
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from src.gs import GaussianParameters
from src.rotations import euler_to_quaternion, quat_to_matrix
from src.features import XFeatMatcher
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

def debug_viz(
    ply_path: str | Path,
    real_image_path: str | Path,
    camera_pose: Sequence[float],
    *,
    scene_pose: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    device: str = "cuda",
    width: int = 1920,
    height: int = 1080,
    fov_deg: float = 60.0,
    near: float = 0.01,
    far: float = 100.0,
    include_world_frame: bool = True,
    axis_length: float | None = None,
    axis_samples: int = 64,
    axis_scale: float = 0.01,
    window_name: str = "FBVS Debug Matches",
    wait_key_ms: int = 0,
    camera=None,
) -> np.ndarray:
    """
    Render a Gaussian-splat view, match features vs. a real image, and show matches.

    Returns:
        BGR match visualization image.
    """
    from src.render_pipeline import render_gaussian_view

    camera_pose_np = np.asarray(camera_pose, dtype=np.float32)
    scene_pose_np = np.asarray(scene_pose, dtype=np.float32)

    real_bgr = cv2.imread(str(real_image_path), cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise FileNotFoundError(f"Could not read image: {real_image_path}")

    result = render_gaussian_view(
        ply_path=ply_path,
        camera_pose=camera_pose_np,
        scene_pose=scene_pose_np,
        device=device,
        width=width,
        height=height,
        fov_deg=fov_deg,
        near=near,
        far=far,
        include_world_frame=include_world_frame,
        axis_length=axis_length,
        axis_samples=axis_samples,
        axis_scale=axis_scale,
        return_alpha=False,
        camera=camera,
    )

    render_rgb_u8 = (np.clip(result.rgb, 0.0, 1.0) * 255).astype(np.uint8)
    render_bgr = cv2.cvtColor(render_rgb_u8, cv2.COLOR_RGB2BGR)

    rendered_gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)
    real_gray = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)

    xfeat_matcher = XFeatMatcher()
    match_result = xfeat_matcher.match(rendered_gray, real_gray)

    match_vis = cv2.drawMatches(
        render_bgr,
        match_result.keypoints_ref,
        real_bgr,
        match_result.keypoints_qry,
        match_result.matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Display (optional blocking)
    shown = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, match_vis)
        shown = True

        if wait_key_ms <= 0:
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        else:
            cv2.waitKey(wait_key_ms)
    except cv2.error:
        pass
    finally:
        if shown:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

    return match_vis


def debug_viz_mesh(
    ply_path: str | Path,
    real_image_path: str | Path,
    camera_pose: np.ndarray,
    *,
    width: int = 1920,
    height: int = 1080,
    fov_deg: float = 60.0,
    near: float = 0.01,
    far: float = 100.0,
    window_name: str = "FBVS Mesh Debug Matches",
    wait_key_ms: int = 0,
    camera=None,
) -> np.ndarray:
    """
    Render a mesh view, then visualize raw XFeat matches against a real image.
    """
    from src.camera import create_default_camera
    from src.mesh_scene import MeshScene

    camera_pose_np = np.asarray(camera_pose, dtype=np.float32)

    scene = MeshScene(verbose=False)
    scene.load_mesh(str(ply_path))

    if camera is None:
        camera = create_default_camera(width=width, height=height, fov_deg=fov_deg)

    camera.set_pose(position=camera_pose_np[:3], orientation=camera_pose_np[3:])

    render_rgb = scene.render_from_virtual_camera(camera, near=near, far=far)
    render_rgb_u8 = (np.clip(render_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    render_bgr = cv2.cvtColor(render_rgb_u8, cv2.COLOR_RGB2BGR)
    real_bgr = cv2.imread(str(real_image_path), cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise FileNotFoundError(f"Could not read image: {real_image_path}")

    rendered_gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)
    real_gray = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)

    xfeat_matcher = XFeatMatcher()
    match_result = xfeat_matcher.match(rendered_gray, real_gray)
    match_vis = cv2.drawMatches(
        render_bgr,
        match_result.keypoints_ref,
        real_bgr,
        match_result.keypoints_qry,
        match_result.matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Display (optional blocking)
    shown = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, match_vis)
        shown = True

        if wait_key_ms <= 0:
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        else:
            cv2.waitKey(wait_key_ms)
    except cv2.error:
        pass
    finally:
        if shown:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

    return match_vis


def pose6_from_T(T: np.ndarray, matrix_type: str ="w2c") -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    if matrix_type == "w2c":
        T_c2w = np.linalg.inv(T)
    elif matrix_type == "c2w":
        T_c2w = T
    else:
        raise ValueError("matrix_type must be 'w2c' or 'c2w'")

    pos = T_c2w[:3, 3]
    rot = R.from_matrix(T_c2w[:3, :3]).as_euler("xyz", degrees=False)
    return np.concatenate([pos,rot]).astype(np.float32)
