from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from src.features import Features
from src.render_pipeline import render_gaussian_view
from src.control import (
    geometric_error,
    normalize_features,
    interaction_matrix,
    velocity,
)
from src.camera import create_camera_from_K
from src.utils_colmap import get_camera_intrinsics


def _compose_current_target_view(current_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Build a side-by-side visualization of current render and target view."""
    h, w = current_bgr.shape[:2]
    if target_bgr.shape[:2] != (h, w):
        target_vis = cv2.resize(target_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        target_vis = target_bgr

    panel = np.hstack([current_bgr, target_vis])
    cv2.line(panel, (w, 0), (w, h), (255, 255, 255), 1)
    return panel


def _draw_iteration_features(
    panel: np.ndarray,
    current_points_px: np.ndarray,
    target_points_px: np.ndarray,
    current_size: tuple[int, int],
    target_size: tuple[int, int],
) -> None:
    """Overlay matched feature points on left(current) and right(target) images."""
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

        # Left side: current rendered features.
        u_cur = int(round(float(current_points_px[i, 0])))
        v_cur = int(round(float(current_points_px[i, 1])))
        if 0 <= u_cur < w and 0 <= v_cur < h:
            cv2.circle(panel, (u_cur, v_cur), 6, color, 2, lineType=cv2.LINE_AA)

        # Right side: desired target features.
        u_tgt = int(round(float(target_points_px[i, 0]) * sx)) + w
        v_tgt = int(round(float(target_points_px[i, 1]) * sy))
        if w <= u_tgt < 2 * w and 0 <= v_tgt < h:
            cv2.circle(panel, (u_tgt, v_tgt), 6, color, 2, lineType=cv2.LINE_AA)


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
    Debug helper:
    1) Render a Gaussian-splat view from a given pose (same flow as main.py),
    2) Match SIFT features against a real image from disk,
    3) Show the match visualization.

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
    kp1, kp2, matches = f.match_sift()
    mask = f.ransac_filter(kp1, kp2, matches)
    inlier_matches = [m for m, keep in zip(matches, mask) if keep]
    _, _, quad_matches = f.extract_quad_features(kp1, kp2, inlier_matches)

    # Draw only the 4 selected matches (with match lines), no area/quad overlay.
    match_vis = cv2.drawMatches(
        render_bgr,
        kp1,
        real_bgr,
        kp2,
        quad_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    shown = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, match_vis)
        shown = True

        if wait_key_ms <= 0:
            # Block for debug usage until user closes window or presses q/esc.
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                try:
                    visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                    if visible < 1:
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


def fbvs(
    scene,
    initial_pose,
    desired_view,
    *,
    max_iters: int = 100,
    error_tolerance: float = 0.01,
    gain: float = 0.1,
    dt: float = 1.0,
    vis_window_name: str = "FBVS Current vs Target",
    vis_wait_ms: int = 1,
):
    camera_pose_np = np.asarray(initial_pose, dtype=np.float32).copy()
    scene_pose_np = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float32)

    real_bgr = cv2.imread(str(desired_view), cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise FileNotFoundError(f"Could not read image: {desired_view}")

    # Initialize scene and camera outside the loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.gs import GaussianSplattingScene
    scene_obj = GaussianSplattingScene(device=str(device))
    scene_obj.load_from_ply(str(scene))

    K = get_camera_intrinsics("data/sparse/0")
    camera = create_camera_from_K(K, width=1920, height=1080)
    cam_h = int(camera.intrinsics.height)
    cam_w = int(camera.intrinsics.width)
    if real_bgr.shape[:2] != (cam_h, cam_w):
        print(
            f"[fbvs] resizing desired view from {real_bgr.shape[1]}x{real_bgr.shape[0]} "
            f"to {cam_w}x{cam_h} to match camera intrinsics."
        )
        real_bgr = cv2.resize(real_bgr, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
    real_gray = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)

    last_rendered_features = None
    last_real_features = None
    converged = False
    tracked_features_px = None
    desired_features_px = None
    desired_features_norm = None
    feature_points_world = None

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

        # Initialize and filter features only once at the beginning.
        if tracked_features_px is None:
            rendered_depth_map = result.depth
            if rendered_depth_map is None:
                print("[fbvs] Failed to extract rendered depth map from Gaussian renderer.")
                break

            f = Features(rendered_gray, real_gray)
            kp1, kp2, matches = f.match_sift()
            mask = f.ransac_filter(kp1, kp2, matches)
            inlier_matches = [m for m, keep in zip(matches, mask) if keep]
            
            if not inlier_matches:
                print("[fbvs] No inlier matches found during initialization.")
                break

            try:
                tracked_features_px, desired_features_px, _ = f.extract_quad_features(
                    kp1, kp2, inlier_matches
                )
            except ValueError as e:
                print(f"[fbvs] Quad feature extraction failed: {e}")
                break

            tracked_features_px = tracked_features_px.astype(np.float32, copy=True)
            desired_features_px = desired_features_px.astype(np.float32, copy=True)
            desired_features_norm = normalize_features(desired_features_px, camera=camera)

            h, w = rendered_depth_map.shape[:2]
            uv = np.round(tracked_features_px).astype(int)
            uv[:, 0] = np.clip(uv[:, 0], 0, w - 1)
            uv[:, 1] = np.clip(uv[:, 1], 0, h - 1)
            init_depths = rendered_depth_map[uv[:, 1], uv[:, 0]]
            init_depths = np.clip(init_depths, 1e-6, None).astype(np.float32)

            camera.set_pose(position=camera_pose_np[:3], orientation=camera_pose_np[3:])
            feature_points_world = camera.back_project_to_world(
                tracked_features_px, init_depths
            ).astype(np.float32)
            
            # Print desired features only once at the beginning
            d_str = " ".join([f"F{j+1}=({desired_features_px[j,0]:.1f},{desired_features_px[j,1]:.1f})" for j in range(4)])
            print(f"[fbvs] Desired Features: {d_str}")
            print("[fbvs] feature initialization done (once)")
        else:
            if feature_points_world is None:
                print("[fbvs] 3D feature points were not initialized.")
                break
            camera.set_pose(position=camera_pose_np[:3], orientation=camera_pose_np[3:])
            tracked_features_px = camera.project_points(feature_points_world).astype(
                np.float32
            )

        if feature_points_world is None:
            print("[fbvs] 3D feature points were not initialized.")
            break
        points_h = np.hstack(
            [feature_points_world, np.ones((feature_points_world.shape[0], 1))]
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
        if not np.all(valid):
            print(
                "[fbvs] Reprojected 3D features became invalid (behind camera or outside image). "
                "Stopping servoing loop."
            )
            break
        feature_depths = np.clip(feature_depths, 1e-6, None)

        rendered_features = normalize_features(tracked_features_px, camera=camera)
        if desired_features_norm is None:
            raise RuntimeError("Desired features not initialized.")
        real_features = desired_features_norm
        error, e_norm = geometric_error(
            desired_features=real_features, current_features=rendered_features
        )
        last_rendered_features = rendered_features
        last_real_features = real_features

        try:
            panel = _compose_current_target_view(render_bgr, real_bgr)
            if desired_features_px is not None:
                _draw_iteration_features(
                    panel=panel,
                    current_points_px=tracked_features_px,
                    target_points_px=desired_features_px,
                    current_size=render_bgr.shape[:2],
                    target_size=real_bgr.shape[:2],
                )
                
            # Print current feature coordinates to terminal only
            c_str = " ".join([f"F{j+1}=({tracked_features_px[j,0]:.1f},{tracked_features_px[j,1]:.1f})" for j in range(4)])
            print(f"iter={i:02d} error={e_norm:.6f} {c_str}")

            cv2.namedWindow(vis_window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(vis_window_name, panel)
            key = cv2.waitKey(vis_wait_ms) & 0xFF
            if key in (27, ord("q")):
                break
        except cv2.error:
            pass

        # Removed redundant print(f"[fbvs] iter={i:02d} error_norm={e_norm:.6f}")
        if e_norm <= error_tolerance:
            converged = True
            break

        Lx = interaction_matrix(rendered_features, feature_depths)
        v = velocity(Lx, error.reshape(-1), gain=gain).reshape(-1)

        camera.set_pose(position=camera_pose_np[:3], orientation=camera_pose_np[3:])
        camera.apply_velocity(v, dt=dt, frame="camera")
        position, orientation = camera.get_pose()
        camera_pose_np = np.concatenate([position, orientation]).astype(np.float32)

    # If the loop finishes (converged or max_iters reached), 
    # keep window open to allow inspection.
    print("[fbvs] Servoing loop finished. Press any key in the visualization window to close.")
    cv2.waitKey(0)

    try:
        cv2.destroyWindow(vis_window_name)
        cv2.waitKey(1)
    except cv2.error:
        pass

    print(
        f"[fbvs] converged={converged} "
        f"final_pose={camera_pose_np.tolist()}"
    )

    if last_rendered_features is None or last_real_features is None:
        raise RuntimeError("FBVS loop did not produce any feature correspondences.")
    return last_rendered_features, last_real_features
