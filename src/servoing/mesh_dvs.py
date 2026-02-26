from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from src.camera import create_camera_from_K
from src.control import (
    PhotometricVisualServoingZN,
    image_gradient,
    normalize_photometric_affine,
)
from src.mesh_scene import MeshScene
from src.servoing.mesh_common import default_mesh_intrinsics, render_view
from src.servoing.target_io import _load_and_fit_target
from src.servoing.visualization import (
    _compose_current_target_view,
    _make_frame_output_dir,
    _make_video_writer,
    _safe_imshow,
    render_rgb_to_u8,
)


def dvs_mesh(
    scene_ply: str | Path,
    initial_pose: np.ndarray,
    desired_view: str | Path,
    *,
    max_iters: int = 20,
    cost_tolerance: float = 1e3,
    error_tolerance: Optional[float] = None,
    gain: float = 4.0,
    use_zero_mean_normalization: bool = True,
    use_gradient_magnitude: bool = True,
    mu: float = 1e-2,
    vc: float = 0.05,
    dt: float = 1.0,
    vis_window_name: str = "DVS Mesh Current vs Target",
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
    Direct Visual Servoing (DVS) loop with ZN Gauss-Newton control.

    Flow per iteration:
      1) render current image I(r_k)
      2) compute e = I(r_k) - I*
      3) compute gradients Ix, Iy
      4) compute dense luminance interaction matrix L_I
      5) compute velocity via ZN/Gauss-Newton update
      6) apply velocity for dt
      7) stop on cost threshold or max iterations
    """
    if error_tolerance is not None:
        # Backward-compatibility alias.
        cost_tolerance = float(error_tolerance)

    camera_pose = np.asarray(initial_pose, dtype=np.float32).reshape(-1).copy()
    if camera_pose.size != 6:
        raise ValueError(f"initial_pose must have 6 elements, got shape {camera_pose.shape}")

    scene_obj = MeshScene(verbose=verbose)
    scene_obj.load_mesh(str(scene_ply))

    K = default_mesh_intrinsics()
    camera = create_camera_from_K(K, width=1296, height=968)
    cam_h = int(camera.intrinsics.height)
    cam_w = int(camera.intrinsics.width)

    real_bgr, real_gray = _load_and_fit_target(desired_view, cam_w, cam_h, verbose=verbose)

    video_writer = None
    enable_video = bool(verbose and save_video)
    if enable_video:
        video_writer, _ = _make_video_writer(video_path, cam_w, cam_h, fps=fps)

    frame_output_dir = None
    saved_frame_count = 0
    enable_frames = bool(verbose and save_frames)
    if enable_frames:
        frame_output_dir = _make_frame_output_dir(frames_dir)
        print(f"[dvs_mesh] Saving debug frames to {frame_output_dir}")

    if depth_device_name is None:
        depth_device_name = "cuda" if torch.cuda.is_available() else "cpu"

    controller = PhotometricVisualServoingZN(
        camera=camera,
        lambda_gain=gain,
        use_zero_mean_normalization=use_zero_mean_normalization,
        use_gradient_magnitude=use_gradient_magnitude,
        verbose=verbose,
    )

    converged = False
    iteration = 0
    cost = np.inf
    last_velocity = np.zeros(6, dtype=np.float32)

    try:
        while iteration < max_iters:
            camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
            render_rgb, render_gray = render_view(camera, scene_obj)
            desired_gray = normalize_photometric_affine(real_gray, render_gray)
            # Depth estimate Z: use provided MoGe model when available, otherwise mesh depth.
            if depth_model is not None:
                depth_map = camera.render_depth_MoGe(
                    render_rgb,
                    model=depth_model,
                    device_name=depth_device_name,
                    resolution_level=int(depth_resolution_level),
                )
            else:
                depth_map = scene_obj.render_depth_from_virtual_camera(camera)

            depth_map = np.asarray(depth_map, dtype=np.float32)
            valid_depth = np.isfinite(depth_map) & (depth_map > 1e-6)
            if np.any(valid_depth):
                fill_depth = float(np.median(depth_map[valid_depth]))
            else:
                fill_depth = 1.0
            depth_map = np.where(valid_depth, depth_map, fill_depth).astype(np.float32)

            Ix, Iy = image_gradient(
                render_gray,
                fx=camera.intrinsics.fx,
                fy=camera.intrinsics.fy,
            )
            Ix_des, Iy_des = image_gradient(
                desired_gray,
                fx=camera.intrinsics.fx,
                fy=camera.intrinsics.fy,
            )

            v_cmd, cost = controller.compute_control_velocity(
                I_current=render_gray,
                I_desired=desired_gray,
                Ix=Ix,
                Iy=Iy,
                Z=depth_map,
                Ix_des=Ix_des,
                Iy_des=Iy_des,
            )
            last_velocity = np.asarray(v_cmd, dtype=np.float32).reshape(6)

            phase = "gauss-newton"
            if verbose:
                print(
                    f"[dvs_mesh] iter={iteration:03d} cost={cost:.6e} "
                    f"phase={phase} |v|={np.linalg.norm(last_velocity):.6f}"
                )

            need_panel = verbose or enable_video or enable_frames
            if need_panel:
                render_rgb_u8 = render_rgb_to_u8(render_rgb)

                render_bgr = cv2.cvtColor(render_rgb_u8, cv2.COLOR_RGB2BGR)
                panel = _compose_current_target_view(render_bgr, real_bgr)
                cv2.putText(
                    panel,
                    f"iter={iteration:03d} cost={cost:.3e} phase={phase}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    panel,
                    f"|v|={np.linalg.norm(last_velocity):.4f}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

                if enable_video and video_writer is not None:
                    video_writer.write(panel)

                if enable_frames and frame_output_dir is not None:
                    frame_path = frame_output_dir / f"frame_{iteration:04d}.png"
                    if cv2.imwrite(str(frame_path), panel):
                        saved_frame_count += 1
                    elif verbose:
                        print(f"[dvs_mesh] Failed to write debug frame: {frame_path}")

                if verbose:
                    _safe_imshow(vis_window_name, panel, vis_wait_ms)

            iteration += 1
            if not np.isfinite(cost):
                if verbose:
                    print("[dvs_mesh] Non-finite cost encountered; stopping.")
                break

            if cost <= cost_tolerance:
                converged = True
                last_velocity = np.zeros(6, dtype=np.float32)
                break

            camera.apply_velocity(last_velocity, dt=dt, frame="camera")
            position, orientation = camera.get_pose()
            camera_pose = np.concatenate([position, orientation]).astype(np.float32)

    finally:
        if enable_video and video_writer is not None:
            video_writer.release()
            if verbose:
                print(f"[dvs_mesh] Simulation saved to {video_path}")

        if enable_frames and frame_output_dir is not None and verbose:
            print(f"[dvs_mesh] Saved {saved_frame_count} debug frames to {frame_output_dir}")

        if verbose:
            try:
                cv2.destroyWindow(vis_window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass

        scene_obj.cleanup()

    print(f"[dvs_mesh] converged={converged}")
    print(f"[dvs_mesh] final_pose={[round(x, 3) for x in camera_pose.flatten()]}")
    if desired_pose is not None:
        desired_pose_arr = np.asarray(desired_pose, dtype=np.float32).reshape(-1)
        if desired_pose_arr.size >= 6:
            diff = desired_pose_arr[:6] - camera_pose[:6]
            print(f"Final Difference: position => {diff[:3]}, rotation => {diff[3:]}")

    metrics = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_cost": float(cost),
        "final_error": float(cost),
        "final_velocity": last_velocity.copy(),
        "open_loop": bool(controller.in_open_loop),
        "cost_history": np.asarray(controller.cost_history, dtype=np.float64).copy(),
    }
    return metrics


__all__ = ["dvs_mesh"]
