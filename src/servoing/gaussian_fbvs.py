from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

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
from src.servoing.visualization import render_rgb_to_u8


def _infer_camera_size_from_K(K: np.ndarray) -> tuple[int, int]:
    """Infer image width/height from principal point when possible."""
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K_arr.shape}")

    width = int(round(float(K_arr[0, 2]) * 2.0))
    height = int(round(float(K_arr[1, 2]) * 2.0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid inferred camera size from K: width={width}, height={height}")
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


def get_features_depth(features: np.ndarray, depth_map: np.ndarray):
    coords = np.asarray(features, dtype=np.float32).astype(int)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"features must have shape (N,2), got {coords.shape}")
    x, y = coords[:, 0], coords[:, 1]
    h, w = depth_map.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return depth_map[y, x]


def fbvs(
    scene: str | Path,
    initial_pose: Sequence[float],
    desired_view: str | Path,
    *,
    max_iters: int = 1000,
    error_tolerance: float = 0.12,
    gain: float = 0.7,
    dt: float = 1.0,
    vis_window_name: str = "FBVS Current vs Target",
    vis_wait_ms: int = 1,
    K: np.ndarray = None,
    save_video: bool = True,
    video_path: str = "output_gs_fbvs.mp4",
    fps: float = 10.0,
    save_frames: bool = True,
    frames_dir: str | Path = "debug_frames",
    verbose: bool = False,
    desired_pose: np.ndarray = None,
):
    camera_pose = np.asarray(initial_pose, dtype=np.float32).copy()
    if camera_pose.size != 6:
        raise ValueError(f"initial_pose must have 6 elements, got shape {camera_pose.shape}")
    scene_pose_np = np.zeros(6, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_obj, camera, cam_w, cam_h = _init_scene_and_camera(scene, K, device)
    real_bgr, real_gray = _load_and_fit_target(desired_view, cam_w, cam_h)

    video_writer = None
    enable_video = bool(verbose and save_video)
    if enable_video:
        video_writer, _ = _make_video_writer(video_path, cam_w, cam_h, fps=fps)

    frame_output_dir = None
    saved_frame_count = 0
    enable_frames = bool(verbose and save_frames)
    if enable_frames:
        frame_output_dir = _make_frame_output_dir(frames_dir)
        print(f"[fbvs_gaussian] Saving debug frames to {frame_output_dir}")

    converged = False
    iteration = 0
    norm = np.inf
    last_rendered_features = None
    last_real_features = None

    try:
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
        render_depth = np.asarray(result.depth, dtype=np.float32)

        xf = XFeatMatcher(top_k=1024)
        match_result = xf.match(render_gray, real_gray)
        kp_ref, kp_qry, matches = (
            match_result.keypoints_ref,
            match_result.keypoints_qry,
            match_result.matches,
        )
        current_features_px = np.array(
            [kp_ref[m.queryIdx].pt for m in matches], dtype=np.float32
        )
        desired_features_px = np.array(
            [kp_qry[m.trainIdx].pt for m in matches], dtype=np.float32
        )
        desired_features_norm = normalize_features(desired_features_px, camera=camera)

        current_features_depths = get_features_depth(current_features_px, render_depth)
        current_features_depths = np.clip(
            np.asarray(current_features_depths, dtype=np.float32), 1e-6, None
        )
        feature_points_world = camera.back_project_to_world(
            current_features_px, current_features_depths
        ).astype(np.float32)
        while norm > error_tolerance and iteration < max_iters:
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
            render_depth = np.asarray(result.depth, dtype=np.float32)

            current_features_px = camera.project_points(feature_points_world).astype(np.float32)

            points_h = np.hstack(
                [
                    feature_points_world,
                    np.ones((feature_points_world.shape[0], 1), dtype=np.float32),
                ]
            )
            points_cam = (camera.get_view_matrix() @ points_h.T).T[:, :3]
            current_features_depths = points_cam[:, 2].astype(np.float32)

            h_img, w_img = render_rgb.shape[:2]
            u = current_features_px[:, 0]
            v = current_features_px[:, 1]
            valid = (
                (current_features_depths > 1e-6)
                & (u >= 0.0)
                & (u < float(w_img))
                & (v >= 0.0)
                & (v < float(h_img))
            )

            if np.sum(valid) < 3:
                if verbose:
                    print(
                        f"[fbvs_gaussian] Too few features visible ({np.sum(valid)}). Stopping servoing loop."
                    )
                break

            need_panel = verbose or enable_video or enable_frames
            if need_panel:
                render_rgb_u8 = render_rgb_to_u8(render_rgb)
                render_bgr = cv2.cvtColor(render_rgb_u8, cv2.COLOR_RGB2BGR)
                panel = _compose_current_target_view(render_bgr, real_bgr)
                _draw_iteration_features(
                    panel=panel,
                    current_points_px=current_features_px,
                    target_points_px=desired_features_px,
                    current_size=render_bgr.shape[:2],
                    target_size=real_bgr.shape[:2],
                )

                if enable_video and video_writer is not None:
                    video_writer.write(panel)

                if enable_frames and frame_output_dir is not None:
                    frame_path = frame_output_dir / f"frame_{iteration:04d}.png"
                    if cv2.imwrite(str(frame_path), panel):
                        saved_frame_count += 1
                    elif verbose:
                        print(f"[fbvs_gaussian] Failed to write debug frame: {frame_path}")

                if verbose:
                    _safe_imshow(vis_window_name, panel, vis_wait_ms)

            current_features_norm = normalize_features(current_features_px[valid], camera=camera)
            desired_features_norm_valid = desired_features_norm[valid]
            current_features_depths_valid = current_features_depths[valid]

            error, norm = geometric_error(desired_features_norm_valid, current_features_norm)
            last_rendered_features = current_features_norm
            last_real_features = desired_features_norm_valid

            if norm <= error_tolerance:
                converged = True
                break

            L = interaction_matrix(current_features_norm, current_features_depths_valid)
            v_cmd = velocity(L, error.reshape(-1), gain)
            camera.apply_velocity(v_cmd, dt=dt, frame="camera")
            position, orientation = camera.get_pose()
            camera_pose = np.concatenate([position, orientation]).astype(np.float32)

            iteration += 1
            if verbose:
                print(
                    f"[fbvs_gaussian] Iteration: {iteration}, error norm: {norm}, velocity norm: {np.linalg.norm(v_cmd)}"
                )
    finally:
        if enable_video and video_writer is not None:
            video_writer.release()
            if verbose:
                print(f"[fbvs_gaussian] Simulation saved to {video_path}")

        if enable_frames and frame_output_dir is not None and verbose:
            print(
                f"[fbvs_gaussian] Saved {saved_frame_count} debug frames to {frame_output_dir}"
            )

        if verbose:
            try:
                cv2.destroyWindow(vis_window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

        scene_obj.cleanup()

    print(f"[fbvs_gaussian] converged={converged}")
    print(f"[fbvs_gaussian] final_pose={[round(x, 3) for x in camera_pose.flatten()]}")
    if desired_pose is not None:
        desired_pose_arr = np.asarray(desired_pose, dtype=np.float32).reshape(-1)
        if desired_pose_arr.size >= 6:
            diff = desired_pose_arr[:6] - camera_pose[:6]
            print(f"Final Difference: position => {diff[:3]}, rotation => {diff[3:]}")

    if last_rendered_features is None or last_real_features is None:
        raise RuntimeError("FBVS Gaussian loop did not produce any feature correspondences.")

    metrics = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_error": float(norm),
    }
    return metrics



__all__ = ["fbvs", "_init_scene_and_camera", "_init_features_and_world_points"]
