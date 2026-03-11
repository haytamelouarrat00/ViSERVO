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
    normalize_image, ZNSSD,
    image_gradient, PhotometricVisualServoingZN,
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
        video_writer, _ = _make_video_writer(video_path, cam_w * 3 // 2, cam_h, fps=fps)
        # panel is 3× wide (current | desired | error), writer uses 3*cam_w

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
    znssd = ZNSSD(real_gray, render_gray)

    if verbose:
        print('[Gaussian DVS] Initial cost: ', cost, '  ZNSSD: ', znssd)

    # --- PVS controller ---
    pvs = PhotometricVisualServoingZN(
        camera=camera,
        lambda_gain=gain,
        use_zero_mean_normalization=use_zero_mean_normalization,
        use_gradient_magnitude=use_gradient_magnitude,
        verbose=verbose,
    )

    fx = float(camera.intrinsics.fx)
    fy = float(camera.intrinsics.fy)

    # Desired-image gradients (evaluated once at ξ*, used to approximate L̂_Ī(ξ*))
    Ix_des, Iy_des = image_gradient(real_gray, fx=fx, fy=fy)

    try:
        while not converged and iteration < max_iters:
            # 1. Render current view
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

            # 2. Compute current-image gradients
            Ix, Iy = image_gradient(render_gray, fx=fx, fy=fy)

            # 3. Control law: v = -λ L̂⁺_Ī(ξ*) · (Ī(ξ) - Ī(ξ*))
            v_cmd, cost = pvs.compute_control_velocity(
                I_current=render_gray,
                I_desired=real_gray,
                Ix=Ix,
                Iy=Iy,
                Z=render_depth,
                Ix_des=Ix_des,
                Iy_des=Iy_des,
            )

            # Clamp velocity to keep camera inside the scene.
            # Without clamping, large initial errors send the camera outside the scene
            # (all-black renders → zero gradients → control law collapses).
            _t_norm = float(np.linalg.norm(v_cmd[:3]))
            _r_norm = float(np.linalg.norm(v_cmd[3:]))
            _MAX_T = 0.3   # max 30 cm translation per iteration
            _MAX_R = 0.3   # max ~17° rotation per iteration
            if _t_norm > _MAX_T:
                v_cmd[:3] *= _MAX_T / _t_norm
            if _r_norm > _MAX_R:
                v_cmd[3:] *= _MAX_R / _r_norm

            znssd = ZNSSD(real_gray, render_gray)

            # 4. Visualization
            need_panel = verbose or enable_video or enable_frames
            if need_panel:
                render_rgb_u8 = np.clip(render_rgb * 255.0, 0, 255).astype(np.uint8)
                render_bgr_vis = cv2.cvtColor(render_rgb_u8, cv2.COLOR_RGB2BGR)
                panel = _compose_current_target_view(render_bgr_vis, real_bgr)

                # Photometric error map: visually confirms camera movement even for small steps
                ph_diff = render_gray - real_gray
                err_vis = np.clip(
                    (ph_diff - ph_diff.min()) / (ph_diff.max() - ph_diff.min() + 1e-8) * 255.0,
                    0, 255,
                ).astype(np.uint8)
                err_color = cv2.applyColorMap(err_vis, cv2.COLORMAP_JET)
                err_color = cv2.resize(err_color, (render_bgr_vis.shape[1], render_bgr_vis.shape[0]))
                panel = np.hstack([panel, err_color])

                # Iteration / pose / cost text overlay
                v_norm = float(np.linalg.norm(v_cmd))
                cv2.putText(panel, f"iter={iteration}  cost={cost:.2e}  ZNSSD={znssd:.4f}  ||v||={v_norm:.4f}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, f"pose={[round(float(x), 3) for x in camera_pose]}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                if enable_video and video_writer is not None:
                    video_writer.write(panel)
                if enable_frames and frame_output_dir is not None:
                    frame_path = frame_output_dir / f"frame_{iteration:04d}.png"
                    if cv2.imwrite(str(frame_path), panel):
                        saved_frame_count += 1
                if verbose:
                    _safe_imshow(vis_window_name, panel, vis_wait_ms)

            if verbose:
                print(
                    f"[Gaussian DVS] Iter {iteration}: "
                    f"cost={cost:.4f}  ZNSSD={znssd:.4f}  ||v||={np.linalg.norm(v_cmd):.6f}  "
                    f"pose={[round(float(x), 3) for x in camera_pose]}"
                )

            # 5. Check convergence
            if cost < cost_tolerance:
                converged = True
                break
            if error_tolerance is not None and znssd < error_tolerance:
                converged = True
                break

            # 6. Apply velocity and update pose
            camera.apply_velocity(v_cmd, dt=dt, frame="camera")
            position, orientation = camera.get_pose()
            camera_pose = np.concatenate([position, orientation]).astype(np.float32)

            iteration += 1

    finally:
        if enable_video and video_writer is not None:
            video_writer.release()
            if verbose:
                print(f"[Gaussian DVS] Video saved to {video_path}")
        if enable_frames and frame_output_dir is not None and verbose:
            print(f"[Gaussian DVS] Saved {saved_frame_count} debug frames to {frame_output_dir}")
        if verbose:
            try:
                cv2.destroyWindow(vis_window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass
        scene_obj.cleanup()

    print(f"[Gaussian DVS] converged={converged}")
    print(f"[Gaussian DVS] final_pose={[round(x, 3) for x in camera_pose.flatten()]}")
    if desired_pose is not None:
        desired_pose_arr = np.asarray(desired_pose, dtype=np.float32).reshape(-1)
        if desired_pose_arr.size >= 6:
            diff = desired_pose_arr[:6] - camera_pose[:6]
            print(f"Final Difference: position => {diff[:3]}, rotation => {diff[3:]}")

    final_cost = float(pvs.cost_history[-1]) if pvs.cost_history else float("nan")
    return {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_cost": final_cost,
    }
