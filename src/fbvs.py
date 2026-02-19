from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch

from src.camera import create_camera_from_K, VirtualCamera
from src.control import (
    geometric_error,
    interaction_matrix,
    normalize_features,
    velocity,
)
from src.features import XFeatMatcher
from src.mesh_scene import MeshScene
from src.render_pipeline import render_gaussian_view


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def _compose_current_target_view(
    current_bgr: np.ndarray, target_bgr: np.ndarray
) -> np.ndarray:
    """Return a side-by-side panel: left=current render, right=target image."""
    h, w = current_bgr.shape[:2]
    if target_bgr.shape[:2] != (h, w):
        target_bgr = cv2.resize(target_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    panel = np.hstack([current_bgr, target_bgr])
    cv2.line(panel, (w, 0), (w, h), (255, 255, 255), 1)
    return panel


def _draw_iteration_features(
    panel: np.ndarray,
    current_points_px: np.ndarray,
    target_points_px: np.ndarray,
    current_size: tuple[int, int],
    target_size: tuple[int, int],
) -> None:
    """Overlay matched feature points on left(current) and right(target) halves."""
    h, w = current_size
    target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        return

    sx = float(w) / float(target_w)
    sy = float(h) / float(target_h)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

    n = min(len(current_points_px), len(target_points_px))
    for i in range(n):
        color = colors[i % len(colors)]

        # Left: current
        u_cur = int(round(float(current_points_px[i, 0])))
        v_cur = int(round(float(current_points_px[i, 1])))
        if 0 <= u_cur < w and 0 <= v_cur < h:
            cv2.circle(panel, (u_cur, v_cur), 6, color, 2, lineType=cv2.LINE_AA)

        # Right: target (scaled) + x-offset by w
        u_tgt = int(round(float(target_points_px[i, 0]) * sx)) + w
        v_tgt = int(round(float(target_points_px[i, 1]) * sy))
        if w <= u_tgt < 2 * w and 0 <= v_tgt < h:
            cv2.circle(panel, (u_tgt, v_tgt), 6, color, 2, lineType=cv2.LINE_AA)


def _safe_imshow(window_name: str, img: np.ndarray, wait_ms: int) -> None:
    """Best-effort imshow that won't crash headless environments."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(wait_ms)
    except cv2.error:
        pass


# -----------------------------------------------------------------------------
# Debug visualization (single render + matches)
# -----------------------------------------------------------------------------


def _init_scene_and_camera(
    scene_path: str | Path, K:np.ndarray, device: torch.device
):
    """Load Gaussian scene and camera intrinsics."""
    from src.gs import GaussianSplattingScene

    scene_obj = GaussianSplattingScene(device=str(device))
    scene_obj.load_from_ply(str(scene_path))
    
    camera = create_camera_from_K(K, width=1920, height=1080)

    cam_h = int(camera.intrinsics.height)
    cam_w = int(camera.intrinsics.width)
    return scene_obj, camera, cam_w, cam_h


def _load_and_fit_target(
    desired_view_path: str | Path, cam_w: int, cam_h: int, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Load desired view and resize to camera resolution; return (bgr, gray)."""
    real_bgr = cv2.imread(str(desired_view_path), cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise FileNotFoundError(f"Could not read image: {desired_view_path}")

    if real_bgr.shape[:2] != (cam_h, cam_w):
        if verbose:
            print(
                f"[fbvs] resizing desired view from {real_bgr.shape[1]}x{real_bgr.shape[0]} "
                f"to {cam_w}x{cam_h} to match camera intrinsics."
            )
        real_bgr = cv2.resize(real_bgr, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)

    return real_bgr, cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)


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


def _make_video_writer(path: str, cam_w: int, cam_h: int, fps: float = 10.0):
    """Create MP4 writer for side-by-side panel."""
    size = (2 * cam_w, cam_h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer, size


def _make_frame_output_dir(root: str | Path) -> Path:
    """Create a timestamped subdirectory for per-frame debug dumps."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    """
    Feature-Based Visual Servoing (FBVS) loop using Gaussian Splatting.
    """
    camera_pose = np.asarray(initial_pose, dtype=np.float32).copy()
    scene_pose_np = np.zeros(6, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_obj, camera, cam_w, cam_h = _init_scene_and_camera(scene, K, device)
    real_bgr, real_gray = _load_and_fit_target(desired_view, cam_w, cam_h)

    video_writer = None
    if save_video:
        video_writer, _ = _make_video_writer(video_path, cam_w, cam_h, fps=fps)

    frame_output_dir = None
    saved_frame_count = 0
    if save_frames:
        frame_output_dir = _make_frame_output_dir(frames_dir)
        print(f"[fbvs] Saving debug frames to {frame_output_dir}")

    last_rendered_features = None
    last_real_features = None
    converged = False

    tracked_features_px = None
    desired_features_px = None
    desired_features_norm = None
    feature_points_world = None

    try:
        for i in range(max_iters):
            need_depth = feature_points_world is None

            result = render_gaussian_view(
                scene=scene_obj,
                camera_pose=camera_pose,
                scene_pose=scene_pose_np,
                device=str(device),
                camera=camera,
                return_depth=need_depth,
            )

            render_rgb_u8 = (np.clip(result.rgb, 0.0, 1.0) * 255).astype(np.uint8)
            render_bgr = cv2.cvtColor(render_rgb_u8, cv2.COLOR_RGB2BGR)
            rendered_gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)

            # --- one-time feature init ---
            if tracked_features_px is None:
                if result.depth is None:
                    raise RuntimeError(
                        "[fbvs] Failed to extract rendered depth map from Gaussian renderer."
                    )

                (
                    tracked_features_px,
                    desired_features_px,
                    desired_features_norm,
                    feature_points_world,
                ) = _init_features_and_world_points(
                    rendered_gray=rendered_gray,
                    rendered_depth=result.depth,
                    real_gray=real_gray,
                    camera=camera,
                    camera_pose=camera_pose,
                )

                n_feat = len(desired_features_px)
                print(f"[fbvs] Initialized with {n_feat} features.")
                print("[fbvs] feature initialization done (once)")

            # --- project world points to get current tracked px ---
            if feature_points_world is None:
                print("[fbvs] 3D feature points were not initialized.")
                break

            camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
            tracked_features_px = camera.project_points(feature_points_world).astype(
                np.float32
            )

            # --- compute per-feature depth in camera frame ---
            points_h = np.hstack(
                [
                    feature_points_world,
                    np.ones((feature_points_world.shape[0], 1), dtype=np.float32),
                ]
            )
            points_cam = (camera.get_view_matrix() @ points_h.T).T[:, :3]
            feature_depths = points_cam[:, 2].astype(np.float32)

            h_img, w_img = render_bgr.shape[:2]
            u = tracked_features_px[:, 0]
            v = tracked_features_px[:, 1]
            valid = (
                (feature_depths > 1e-6)
                & (u >= 0.0)
                & (u < float(w_img))
                & (v >= 0.0)
                & (v < float(h_img))
            )

            if np.sum(valid) < 3:
                print(
                    f"[fbvs] Too few features visible ({np.sum(valid)}). Stopping servoing loop."
                )
                break

            feature_depths_valid = feature_depths[valid]
            tracked_px_valid = tracked_features_px[valid]
            desired_norm_valid = desired_features_norm[valid]

            rendered_features = normalize_features(tracked_px_valid, camera=camera)
            real_features = desired_norm_valid

            error, e_norm = geometric_error(
                desired_features=real_features, current_features=rendered_features
            )

            last_rendered_features = rendered_features
            last_real_features = real_features

            # --- visualization + logging ---
            panel = _compose_current_target_view(render_bgr, real_bgr)
            if desired_features_px is not None:
                _draw_iteration_features(
                    panel=panel,
                    current_points_px=tracked_features_px,
                    target_points_px=desired_features_px,
                    current_size=render_bgr.shape[:2],
                    target_size=real_bgr.shape[:2],
                )

            print(f"iter={i:02d} error={e_norm:.6f}")

            if video_writer is not None:
                video_writer.write(panel)

            if frame_output_dir is not None:
                frame_path = frame_output_dir / f"frame_{i:04d}.png"
                if cv2.imwrite(str(frame_path), panel):
                    saved_frame_count += 1
                else:
                    print(f"[fbvs] Failed to write debug frame: {frame_path}")

            _safe_imshow(vis_window_name, panel, vis_wait_ms)

            # --- convergence check ---
            if e_norm <= error_tolerance:
                converged = True
                break

            # --- control law: error -> interaction matrix -> velocity -> pose update ---
            Lx = interaction_matrix(rendered_features, feature_depths_valid)
            v_cmd = velocity(Lx, error.reshape(-1), gain=gain).reshape(-1)

            # integrate pose update in camera frame
            camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
            camera.apply_velocity(v_cmd, dt=dt, frame="camera")
            position, orientation = camera.get_pose()
            camera_pose = np.concatenate([position, orientation]).astype(np.float32)

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[fbvs] Simulation saved to {video_path}")

        if frame_output_dir is not None:
            print(
                f"[fbvs] Saved {saved_frame_count} debug frames to {frame_output_dir}"
            )

        try:
            cv2.destroyWindow(vis_window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

    print(f"[fbvs] converged={converged}")
    print(f"[fbvs] final_pose={[round(x, 2) for x in camera_pose.flatten()]}")

    if last_rendered_features is None or last_real_features is None:
        raise RuntimeError("FBVS loop did not produce any feature correspondences.")

    return last_rendered_features, last_real_features


def render_view(camera:VirtualCamera,
                scene: MeshScene):
    rgb = scene.render_from_virtual_camera(camera)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray

def get_features_depth(features: np.ndarray, depth_map: np.ndarray):
    coords = np.array(features).astype(int)
    x, y = coords[:, 0], coords[:, 1]
    h, w = depth_map.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return depth_map[y, x]

def fbvs_mesh(
    scene_ply: str | Path,
    initial_pose: np.ndarray,
    desired_view: str | Path,
    *,
    max_iters: int = 20,
    error_tolerance: float = 0.12,
    gain: float = 0.5,
    dt: float = 1.0,
    vis_window_name: str = "FBVS Mesh Current vs Target",
    vis_wait_ms: int = 1,
    save_video: bool = False,
    video_path: str = "output_mesh.mp4",
    fps: float = 10.0,
    save_frames: bool = False,
    frames_dir: str | Path = "debug_frames_mesh",
    verbose: bool = False,
    desired_pose: np.ndarray = None,
    depth_model: Optional[Any] = None,
    depth_device_name: Optional[str] = None,
    depth_resolution_level: int = 6,

):
    """
    Feature-Based Visual Servoing (FBVS) loop using a Triangle Mesh scene.
    """
    from src.mesh_scene import MeshScene

    camera_pose = np.asarray(initial_pose, dtype=np.float32).copy()

    # Init mesh scene
    scene_obj = MeshScene(verbose=verbose)
    scene_obj.load_mesh(str(scene_ply))

    # Init camera
    K = np.array([
        [1158.3, 0, 649],
        [0, 1153.53, 483.5],
        [0, 0, 1]
    ])
    camera = create_camera_from_K(K, width=1296, height=968)
    cam_h = int(camera.intrinsics.height)
    cam_w = int(camera.intrinsics.width)

    real_bgr, real_gray = _load_and_fit_target(
        desired_view, cam_w, cam_h, verbose=verbose
    )

    video_writer = None
    enable_video = bool(verbose and save_video)
    if enable_video:
        video_writer, _ = _make_video_writer(video_path, cam_w, cam_h, fps=fps)

    frame_output_dir = None
    saved_frame_count = 0
    enable_frames = bool(verbose and save_frames)
    if enable_frames:
        frame_output_dir = _make_frame_output_dir(frames_dir)
        print(f"[fbvs_mesh] Saving debug frames to {frame_output_dir}")

    last_rendered_features = None
    last_real_features = None
    converged = False

    tracked_features_px = None
    desired_features_px = None
    desired_features_norm = None
    feature_points_world = None

    camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
    render_rgb, render_gray = render_view(camera, scene_obj)
    if depth_device_name is None:
        depth_device_name = "cuda" if torch.cuda.is_available() else "cpu"
    render_depth = camera.render_depth_MoGe(
        render_rgb,
        model=depth_model,
        device_name=depth_device_name,
        resolution_level=int(depth_resolution_level),
    )

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

    iteration = 0
    norm = np.inf
    while norm > error_tolerance and iteration < max_iters:
        # Reproject fixed 3D feature points into the current view.
        camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
        render_rgb, render_gray = render_view(camera, scene_obj)
        current_features_px = camera.project_points(feature_points_world).astype(np.float32)

        points_h = np.hstack(
            [
                feature_points_world,
                np.ones((feature_points_world.shape[0], 1), dtype=np.float32),
            ]
        )
        points_cam = (camera.get_view_matrix() @ points_h.T).T[:, :3]
        current_features_depths = points_cam[:, 2].astype(np.float32)

        h_img, w_img = render_gray.shape[:2]
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
                    f"[fbvs_mesh] Too few features visible ({np.sum(valid)}). Stopping servoing loop."
                )
            break

        need_panel = verbose
        if need_panel:
            if np.issubdtype(render_rgb.dtype, np.floating):
                if float(np.max(render_rgb)) <= 1.0:
                    render_rgb_u8 = np.clip(render_rgb * 255.0, 0.0, 255.0).astype(
                        np.uint8
                    )
                else:
                    render_rgb_u8 = np.clip(render_rgb, 0.0, 255.0).astype(np.uint8)
            elif render_rgb.dtype == np.uint8:
                render_rgb_u8 = render_rgb
            else:
                render_rgb_u8 = np.clip(render_rgb, 0, 255).astype(np.uint8)

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
                else:
                    print(f"[fbvs_mesh] Failed to write debug frame: {frame_path}")

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
                f"[fbvs_mesh] Iteration: {iteration}, error norm: {norm}, velocity norm: {np.linalg.norm(v_cmd)}"
            )

    if enable_video and video_writer is not None:
        video_writer.release()
        print(f"[fbvs_mesh] Simulation saved to {video_path}")

    if enable_frames and frame_output_dir is not None:
        print(f"[fbvs_mesh] Saved {saved_frame_count} debug frames to {frame_output_dir}")

    if verbose:
        try:
            cv2.destroyWindow(vis_window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

    print(f"[fbvs_mesh] converged={converged}")
    print(f"[fbvs_mesh] final_pose={[round(x, 2) for x in camera_pose.flatten()]}")
    if desired_pose is not None:
        diff = desired_pose - camera_pose
        pos = diff[:3]
        rot = diff[3:]
        print(f"Final Difference: position => {pos}, rotation => {rot}")

    if last_rendered_features is None or last_real_features is None:
        raise RuntimeError("FBVS mesh loop did not produce any feature correspondences.")

    metrics = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_error": float(norm),
    }
    scene_obj.cleanup()
    return metrics
