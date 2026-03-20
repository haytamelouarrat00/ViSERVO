from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
from src.features import XFeatMatcher
from src.mesh_scene import MeshScene
from src.servoing.mesh_common import (
    default_mesh_intrinsics,
    get_features_depth,
    render_view,
)
from src.servoing.target_io import _load_and_fit_target
from src.servoing.visualization import (
    _compose_current_target_view,
    _draw_iteration_features,
    _make_frame_output_dir,
    _make_video_writer,
    _safe_imshow,
    render_rgb_to_u8,
)


def fbvs_mesh(
    scene_ply: str | Path,
    initial_pose: np.ndarray,
    desired_view: str | Path,
    *,
    max_iters: int = 100,
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
    camera_pose = np.asarray(initial_pose, dtype=np.float32).copy()

    # Init mesh scene
    scene_obj = MeshScene(verbose=verbose)
    scene_obj.load_mesh(str(scene_ply))

    # Init camera
    K = default_mesh_intrinsics()
    camera = create_camera_from_K(K, width=1296, height=968)
    cam_h = int(camera.intrinsics.height)
    cam_w = int(camera.intrinsics.width)

    real_bgr, real_gray = _load_and_fit_target(
        desired_view, cam_w, cam_h, verbose=verbose
    )

    video_writer = None
    enable_video = bool(save_video)
    if enable_video:
        video_writer, _ = _make_video_writer(video_path, cam_w, cam_h, fps=fps)

    frame_output_dir = None
    saved_frame_count = 0
    enable_frames = bool(save_frames)
    if enable_frames:
        frame_output_dir = _make_frame_output_dir(frames_dir)
        print(f"[fbvs_mesh] Saving debug frames to {frame_output_dir}")

    last_rendered_features = None
    last_real_features = None
    converged = False

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
        current_features_px = camera.project_points(feature_points_world).astype(
            np.float32
        )

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
                else:
                    print(f"[fbvs_mesh] Failed to write debug frame: {frame_path}")

            if verbose:
                _safe_imshow(vis_window_name, panel, vis_wait_ms)

        current_features_norm = normalize_features(
            current_features_px[valid], camera=camera
        )
        desired_features_norm_valid = desired_features_norm[valid]
        current_features_depths_valid = current_features_depths[valid]

        error, norm = geometric_error(
            desired_features_norm_valid, current_features_norm
        )
        last_rendered_features = current_features_norm
        last_real_features = desired_features_norm_valid

        if norm <= error_tolerance:
            converged = True
            break

        L = interaction_matrix(current_features_norm, current_features_depths_valid)
        v_cmd = velocity(L, error.reshape(-1), gain)

        # Clamp velocity to prevent leaving scene bounds on large initial errors
        t_norm = float(np.linalg.norm(v_cmd[:3]))
        r_norm = float(np.linalg.norm(v_cmd[3:]))
        MAX_T, MAX_R = 0.3, 0.3
        if t_norm > MAX_T:
            v_cmd[:3] *= MAX_T / t_norm
        if r_norm > MAX_R:
            v_cmd[3:] *= MAX_R / r_norm

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
        print(
            f"[fbvs_mesh] Saved {saved_frame_count} debug frames to {frame_output_dir}"
        )

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

    metrics = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_error": float(norm),
    }
    scene_obj.cleanup()
    return metrics


def fbvs_mesh_redetect(
    scene_ply: str | Path,
    initial_pose: np.ndarray,
    desired_view: str | Path,
    *,
    max_iters: int = 100,
    error_tolerance: float = 0.35,
    gain: float = 1.0,
    dt: float = 1.0,
    vis_window_name: str = "FBVS Mesh Redetect Current vs Target",
    vis_wait_ms: int = 1,
    save_video: bool = False,
    video_path: str = "output_mesh_redetect.mp4",
    fps: float = 30.0,
    save_frames: bool = False,
    frames_dir: str | Path = "debug_frames_mesh_redetect",
    verbose: bool = False,
    desired_pose: np.ndarray = None,
    depth_model: Optional[Any] = None,
    depth_device_name: Optional[str] = None,
    depth_resolution_level: int = 6,
):
    """
    Feature-Based Visual Servoing (FBVS) using a Triangle Mesh scene with dynamic feature re-detection.

    This version re-detects and re-matches features every iteration, providing better robustness
    to large displacements compared to static feature tracking.
    """
    camera_pose = np.asarray(initial_pose, dtype=np.float32).copy()

    # Init mesh scene
    scene_obj = MeshScene(verbose=verbose)
    scene_obj.load_mesh(str(scene_ply))

    # Init camera
    K = default_mesh_intrinsics()
    camera = create_camera_from_K(K, width=1296, height=968)
    cam_h = int(camera.intrinsics.height)
    cam_w = int(camera.intrinsics.width)

    real_bgr, real_gray = _load_and_fit_target(
        desired_view, cam_w, cam_h, verbose=verbose
    )

    video_writer = None
    enable_video = bool(save_video)
    if enable_video:
        video_writer, _ = _make_video_writer(video_path, cam_w, cam_h, fps=fps)

    frame_output_dir = None
    saved_frame_count = 0
    enable_frames = bool(save_frames)
    if enable_frames:
        frame_output_dir = _make_frame_output_dir(frames_dir)
        print(f"[fbvs_mesh_redetect] Saving debug frames to {frame_output_dir}")

    converged = False
    iteration = 0
    norm = np.inf
    last_rendered_features = None
    last_real_features = None
    xf = XFeatMatcher(top_k=1024)

    try:
        while norm > error_tolerance and iteration < max_iters:
            camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
            render_rgb, render_gray = render_view(camera, scene_obj)

            # Per-iteration mesh depth — exact and fast, avoids the stale-depth bug
            # that results from sampling a depth map estimated at a different camera pose.
            # Background pixels (no mesh hit) have depth=0; fill with the median of
            # valid pixels so features on the border still get a usable depth value.
            render_depth = np.asarray(
                scene_obj.render_depth_from_virtual_camera(camera), dtype=np.float32
            )
            _valid_d = np.isfinite(render_depth) & (render_depth > 1e-6)
            _fill = float(np.median(render_depth[_valid_d])) if np.any(_valid_d) else 1.0
            render_depth = np.where(_valid_d, render_depth, _fill).astype(np.float32)

            match_result = xf.match(render_gray, real_gray)
            kp_ref, kp_qry, matches = (
                match_result.keypoints_ref,
                match_result.keypoints_qry,
                match_result.matches,
            )

            if len(matches) < 3:
                if verbose:
                    print(
                        f"[fbvs_mesh_redetect] Too few matched features ({len(matches)}). Stopping servoing loop."
                    )
                break

            current_features_px = np.array(
                [kp_ref[m.queryIdx].pt for m in matches], dtype=np.float32
            ).reshape(-1, 2)
            desired_features_px = np.array(
                [kp_qry[m.trainIdx].pt for m in matches], dtype=np.float32
            ).reshape(-1, 2)

            # Sample depth directly from the pre-filled current-pose mesh depth map
            current_features_depths = np.asarray(
                get_features_depth(current_features_px, render_depth), dtype=np.float32
            )
            valid = current_features_depths > 1e-6

            if np.sum(valid) < 3:
                if verbose:
                    print(
                        "[fbvs_mesh_redetect] Too few valid feature depths. Stopping servoing loop."
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
                        print(
                            f"[fbvs_mesh_redetect] Failed to write debug frame: {frame_path}"
                        )

                if verbose:
                    _safe_imshow(vis_window_name, panel, vis_wait_ms)

            current_features_norm = normalize_features(
                current_features_px[valid], camera=camera
            )
            desired_features_norm = normalize_features(
                desired_features_px[valid], camera=camera
            )

            error, norm = geometric_error(desired_features_norm, current_features_norm)
            last_rendered_features = current_features_norm
            last_real_features = desired_features_norm

            if norm <= error_tolerance:
                converged = True
                break

            L = interaction_matrix(current_features_norm, current_features_depths[valid])
            v_cmd = velocity(L, error.reshape(-1), gain)

            # Clamp velocity to prevent leaving scene bounds on large initial errors
            t_norm = float(np.linalg.norm(v_cmd[:3]))
            r_norm = float(np.linalg.norm(v_cmd[3:]))
            MAX_T, MAX_R = 0.3, 0.3
            if t_norm > MAX_T:
                v_cmd[:3] *= MAX_T / t_norm
            if r_norm > MAX_R:
                v_cmd[3:] *= MAX_R / r_norm

            camera.apply_velocity(v_cmd, dt=dt, frame="camera")
            position, orientation = camera.get_pose()
            camera_pose = np.concatenate([position, orientation]).astype(np.float32)

            iteration += 1
            if verbose:
                print(
                    "[fbvs_mesh_redetect] Iteration: "
                    f"{iteration}, error norm: {norm}, velocity norm: {np.linalg.norm(v_cmd)}"
                )
    finally:
        if enable_video and video_writer is not None:
            video_writer.release()
            if verbose:
                print(f"[fbvs_mesh_redetect] Simulation saved to {video_path}")

        if enable_frames and frame_output_dir is not None and verbose:
            print(
                "[fbvs_mesh_redetect] Saved "
                f"{saved_frame_count} debug frames to {frame_output_dir}"
            )

        if verbose:
            try:
                cv2.destroyWindow(vis_window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

        scene_obj.cleanup()

    print(f"[fbvs_mesh_redetect] converged={converged}")
    print(
        f"[fbvs_mesh_redetect] final_pose={[round(x, 3) for x in camera_pose.flatten()]}"
    )
    if desired_pose is not None:
        desired_pose_arr = np.asarray(desired_pose, dtype=np.float32).reshape(-1)
        if desired_pose_arr.size >= 6:
            diff = desired_pose_arr[:6] - camera_pose[:6]
            print(f"Final Difference: position => {diff[:3]}, rotation => {diff[3:]}")

    metrics = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_error": float(norm),
    }
    return metrics


__all__ = ["fbvs_mesh", "fbvs_mesh_redetect"]
