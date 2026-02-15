from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from src.camera import create_camera_from_K
from src.control import (
    geometric_error,
    interaction_matrix,
    normalize_features,
    velocity,
)
from src.features import Features
from src.render_pipeline import render_gaussian_view
from src.utils_colmap import get_camera_intrinsics


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
def debug_viz(
    ply_path: str | Path,
    real_image_path: str | Path,
    camera_pose: Sequence[float],
    *,
    scene_pose: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    device: str = "cuda",
    width: int = 1280,
    height: int = 720,
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

    f = Features(rendered_gray, real_gray)
    kp1, kp2, matches = f.match_xfeat()
    mask = f.ransac_filter(kp1, kp2, matches)
    inlier_matches = [m for m, keep in zip(matches, mask) if keep]
    _, _, quad_matches = f.extract_quad_features(kp1, kp2, inlier_matches)

    match_vis = cv2.drawMatches(
        render_bgr,
        kp1,
        real_bgr,
        kp2,
        quad_matches,
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


# -----------------------------------------------------------------------------
# FBVS (feature-based visual servoing)
# -----------------------------------------------------------------------------
def _init_scene_and_camera(
    scene_path: str | Path, sparse_dir: str, device: torch.device
):
    """Load Gaussian scene and camera intrinsics."""
    from src.gs import GaussianSplattingScene

    scene_obj = GaussianSplattingScene(device=str(device))
    scene_obj.load_from_ply(str(scene_path))

    K = get_camera_intrinsics(sparse_dir)
    camera = create_camera_from_K(K, width=1920, height=1080)

    cam_h = int(camera.intrinsics.height)
    cam_w = int(camera.intrinsics.width)
    return scene_obj, camera, cam_w, cam_h


def _load_and_fit_target(
    desired_view_path: str | Path, cam_w: int, cam_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load desired view and resize to camera resolution; return (bgr, gray)."""
    real_bgr = cv2.imread(str(desired_view_path), cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise FileNotFoundError(f"Could not read image: {desired_view_path}")

    if real_bgr.shape[:2] != (cam_h, cam_w):
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
    camera_pose_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize tracked/desired features from first frame and back-project to world.

    Returns:
        tracked_features_px (N,2)
        desired_features_px (N,2)
        desired_features_norm (N,2)
        feature_points_world (N,3)
    """
    f = Features(rendered_gray, real_gray)
    kp1, kp2, matches = f.match_xfeat()
    mask = f.ransac_filter(kp1, kp2, matches)
    inlier_matches = [m for m, keep in zip(matches, mask) if keep]

    if not inlier_matches:
        raise RuntimeError("[fbvs] No inlier matches found during initialization.")

    tracked_features_px = np.array(
        [kp1[m.queryIdx].pt for m in inlier_matches], dtype=np.float32
    )
    desired_features_px = np.array(
        [kp2[m.trainIdx].pt for m in inlier_matches], dtype=np.float32
    )

    desired_features_norm = normalize_features(desired_features_px, camera=camera)

    # sample depth at tracked pixels
    h, w = rendered_depth.shape[:2]
    uv = np.round(tracked_features_px).astype(int)
    uv[:, 0] = np.clip(uv[:, 0], 0, w - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, h - 1)
    init_depths = rendered_depth[uv[:, 1], uv[:, 0]]
    init_depths = np.clip(init_depths, 1e-6, None).astype(np.float32)

    camera.set_pose(position=camera_pose_np[:3], orientation=camera_pose_np[3:])
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
    sparse_dir: str = "data/sparse/0",
    save_video: bool = True,
    video_path: str = "output.mp4",
    fps: float = 10.0,
    save_frames: bool = True,
    frames_dir: str | Path = "debug_frames",
    accel_stop: float = 1e-6,
    v_stop: float = 1e-9,
):
    """
    Feature-Based Visual Servoing (FBVS) loop.

    Key behavior preserved from your original code:
    - Match features ONCE on first iteration.
    - Use depth from first render to back-project to world points.
    - Reproject same world points each iteration to get current 2D features.
    - Build side-by-side visualization and (optionally) save to mp4 and per-frame PNGs.
    """
    camera_pose_np = np.asarray(initial_pose, dtype=np.float32).copy()
    scene_pose_np = np.zeros(6, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_obj, camera, cam_w, cam_h = _init_scene_and_camera(scene, sparse_dir, device)
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

    v_prev = np.zeros(6, dtype=np.float32)

    try:
        for i in range(max_iters):
            need_depth = feature_points_world is None

            result = render_gaussian_view(
                scene=scene_obj,
                camera_pose=camera_pose_np,
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
                    print(
                        "[fbvs] Failed to extract rendered depth map from Gaussian renderer."
                    )
                    break

                try:
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
                        camera_pose_np=camera_pose_np,
                    )
                except RuntimeError as e:
                    print(str(e))
                    break

                n_feat = len(desired_features_px)
                print(f"[fbvs] Initialized with {n_feat} features.")
                print("[fbvs] feature initialization done (once)")

            # --- project world points to get current tracked px ---
            if feature_points_world is None:
                print("[fbvs] 3D feature points were not initialized.")
                break

            camera.set_pose(position=camera_pose_np[:3], orientation=camera_pose_np[3:])
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

            # --- control law ---
            Lx = interaction_matrix(rendered_features, feature_depths_valid)
            v_cmd = velocity(Lx, error.reshape(-1), gain=gain).reshape(-1)

            v_norm = float(np.linalg.norm(v_cmd))
            accel = float(np.linalg.norm(v_cmd - v_prev))

            if v_norm < v_stop:
                print("[fbvs] Velocity reached zero. Stopping.")
                break
            if i > 0 and accel < accel_stop:
                print(
                    f"[fbvs] Acceleration below threshold ({accel:.2e} < {accel_stop:.2e}). Stopping."
                )
                break

            v_prev = v_cmd.copy()

            # integrate pose update in camera frame
            camera.set_pose(position=camera_pose_np[:3], orientation=camera_pose_np[3:])
            camera.apply_velocity(v_cmd, dt=dt, frame="camera")
            position, orientation = camera.get_pose()
            camera_pose_np = np.concatenate([position, orientation]).astype(np.float32)

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[fbvs] Simulation saved to {video_path}")

        if frame_output_dir is not None:
            print(f"[fbvs] Saved {saved_frame_count} debug frames to {frame_output_dir}")

        try:
            cv2.destroyWindow(vis_window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

    print(f"[fbvs] converged={converged}")
    print(f"[fbvs] final_pose={[round(x, 2) for x in camera_pose_np.flatten()]}")

    if last_rendered_features is None or last_real_features is None:
        raise RuntimeError("FBVS loop did not produce any feature correspondences.")

    return last_rendered_features, last_real_features
