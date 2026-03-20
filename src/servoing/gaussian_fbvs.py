from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from sympy import false

from src.camera import create_camera_from_K
from src.control import (
    geometric_error,
    interaction_matrix,
    normalize_features,
    velocity,
)
from src.features import XFeatMatcher
from src.render_pipeline import render_gaussian_view
from src.servoing.target_io import _load_and_fit_target
from src.servoing.trajectory_viz import TrajectoryVisualizer
from src.servoing.visualization import render_rgb_to_u8


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

    scene_obj = GaussianSplattingScene(device=str(device), verbose=False)
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
    image_id: int | str | Path, path: str | Path = "data/playroom/info/images.txt"
) -> Optional[tuple[np.ndarray, str]]:
    """
    Read a COLMAP image pose and return (pose, image_name).

    COLMAP stores image poses as world-to-camera (w2c):
        x_cam = R_w2c * x_world + t_w2c
    This function converts them to camera-to-world (c2w) pose:
        [x, y, z, rx, ry, rz] where xyz is camera center in world and
        r* are XYZ Euler angles of R_c2w.
    """
    # Accept either a numeric ID or an image filename / path
    match_by_name = not str(image_id).lstrip("-").isdigit()
    target_name = Path(image_id).name if match_by_name else None

    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if match_by_name:
            if parts[-1] != target_name:
                continue
        else:
            if int(parts[0]) != int(image_id):
                continue

        image_name = parts[-1]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])

        # COLMAP: world -> camera
        R_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tx, ty, tz], dtype=np.float64)

        # Convert to camera -> world
        R_c2w = R_w2c.T
        C_world = -R_c2w @ t_w2c
        euler_c2w = R.from_matrix(R_c2w).as_euler("xyz")

        pose = np.array(
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
        return pose, image_name

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
    K: np.ndarray = None,
    verbose: bool = False,
    desired_pose: np.ndarray = None,
):
    camera_pose = np.asarray(initial_pose, dtype=np.float32).copy()
    if camera_pose.size != 6:
        raise ValueError(
            f"initial_pose must have 6 elements, got shape {camera_pose.shape}"
        )
    scene_pose_np = np.zeros(6, dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_obj, camera, cam_w, cam_h = _init_scene_and_camera(scene, K, device)
    real_bgr, real_gray = _load_and_fit_target(desired_view, cam_w, cam_h)

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
                break

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
    finally:
        scene_obj.cleanup()

    if last_rendered_features is None or last_real_features is None:
        raise RuntimeError(
            "FBVS Gaussian loop did not produce any feature correspondences."
        )

    metrics = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_error": float(norm),
    }
    return metrics


def fbvs_redetect(
    scene: str | Path,
    initial_pose: Sequence[float],
    desired_view: str | Path,
    *,
    max_iters: int = 100,
    error_tolerance: float = 0.025,
    gain: float = 1,
    dt: float = 1.0,
    K: np.ndarray = None,
    verbose: bool = False,
    desired_pose: np.ndarray = None,
):
    """
    Gaussian FBVS loop that re-detects and re-matches features every iteration.
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

    converged = False
    iteration = 0
    norm = np.inf
    last_rendered_features = None
    last_real_features = None
    xf = XFeatMatcher(top_k=1024)

    try:
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
            render_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)

            match_result = xf.match(render_gray, real_gray)
            kp_ref, kp_qry, matches = (
                match_result.keypoints_ref,
                match_result.keypoints_qry,
                match_result.matches,
            )

            if len(matches) < 3:
                break

            current_features_px = np.array(
                [kp_ref[m.queryIdx].pt for m in matches], dtype=np.float32
            ).reshape(-1, 2)
            desired_features_px = np.array(
                [kp_qry[m.trainIdx].pt for m in matches], dtype=np.float32
            ).reshape(-1, 2)
            current_features_depths = get_features_depth(
                current_features_px, render_depth
            )
            current_features_depths = np.asarray(
                current_features_depths, dtype=np.float32
            )
            valid = current_features_depths > 1e-6

            if np.sum(valid) < 3:
                break

            current_features_norm = normalize_features(
                current_features_px[valid], camera=camera
            )
            desired_features_norm = normalize_features(
                desired_features_px[valid], camera=camera
            )
            current_features_depths_valid = current_features_depths[valid]

            error, norm = geometric_error(desired_features_norm, current_features_norm)
            last_rendered_features = current_features_norm
            last_real_features = desired_features_norm

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
    finally:
        scene_obj.cleanup()

    if last_rendered_features is None or last_real_features is None:
        raise RuntimeError(
            "FBVS Gaussian redetect loop did not produce any feature correspondences."
        )

    metrics = {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_error": float(norm),
    }
    return metrics


fbvs_gaussian_redetect = fbvs_redetect


def _load_colmap_registered_frames(
    colmap_model_path: str | Path,
    images_dir: str | Path,
) -> list[tuple[Path, np.ndarray]]:
    """
    Read a COLMAP model (binary or text) and return all registered frames
    as a list of (image_path, pose_6d) sorted by image filename.

    Poses are returned in the GS/COLMAP world coordinate system,
    so they can be used directly with a GS scene trained from this model.
    """
    import pycolmap
    from src.scene_ops import pose6_from_T

    reconstruction = pycolmap.Reconstruction(str(colmap_model_path))
    images_dir = Path(images_dir)

    frames = []
    for img in reconstruction.images.values():
        image_path = images_dir / img.name
        cam_from_world = img.cam_from_world()
        R_w2c = cam_from_world.rotation.matrix()
        t_w2c = np.array(cam_from_world.translation, dtype=np.float64)
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c
        pose = pose6_from_T(T_w2c, "w2c")
        frames.append((image_path, pose))

    frames.sort(key=lambda x: x[0].name)
    return frames



def plan_trajectory(
    start_id: int,
    end_id: int,
    scene: str | Path,
    colmap_model_path: str | Path,
    images_dir: str | Path,
    K: np.ndarray,
    *,
    func=None,
    error_tolerance: float = 0.05,
    max_iters: int = 50,
    verbose: bool = False,
    stride: int = 1
) -> list[dict]:
    """
    Sequential visual servoing through COLMAP-registered frames.

    `start_id` and `end_id` are indices into the list of registered frames
    sorted by filename (0 = first registered frame alphabetically).

    For each consecutive pair (i, i+1):
      - Image and pose are loaded from the COLMAP model (already in GS coordinates).
      - Servoing starts from the current converged pose.
      - Stops early if a step does not converge.
      - Final rendered frame of each step saved to
        debug_frames/<YYYYMMDD_HHMMSS>/task_<i>_<i+1>.png (all in one flat folder).

    Returns a list of per-step metrics dicts.
    """
    import csv
    import datetime
    from tqdm import tqdm

    if func is None:
        func = fbvs

    frames = _load_colmap_registered_frames(colmap_model_path, images_dir)
    n = len(frames)
    if start_id < 0 or end_id >= n or start_id >= end_id or end_id + stride > n:
        raise ValueError(
            f"start_id={start_id} / end_id={end_id} out of range for {n} registered frames."
        )

    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("debug_frames") / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_pose_np = np.zeros(6, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_pose = frames[start_id][1]

    gt_poses = [frames[j][1] for j in range(start_id, end_id + 1, stride)]

    viz = TrajectoryVisualizer(title=f"Trajectory  {start_id} → {end_id}")
    viz.set_ground_truth(gt_poses)
    viz.add_pose(current_pose, label=str(start_id))

    csv_path = output_dir / "metrics.csv"
    csv_fields = [
        "step", "from", "to",
        "converged", "iterations", "servo_error",
        "euclidean_dist", "geodesic_dist_rad",
    ]

    total_steps = len(range(start_id, end_id, stride))
    all_metrics = []
    try:
        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
            writer.writeheader()
            csv_f.flush()

            pbar = tqdm(range(start_id, end_id, stride), total=total_steps, desc="Trajectory", unit="step")
            for i in pbar:
                target_image, target_pose = frames[i + stride]
                pbar.set_postfix({"step": f"{i}→{i + stride}", "src": frames[i][0].name})

                metrics = func(
                    scene=scene,
                    initial_pose=current_pose,
                    desired_view=target_image,
                    K=K,
                    desired_pose=target_pose,
                    error_tolerance=error_tolerance,
                    max_iters=max_iters,
                    verbose=verbose,
                )

                final_pose = metrics["final_pose"]
                scene_obj, camera, _, _ = _init_scene_and_camera(scene, K, device)
                try:
                    camera.set_pose(position=final_pose[:3], orientation=final_pose[3:])
                    result = render_gaussian_view(
                        scene=scene_obj,
                        camera_pose=final_pose,
                        scene_pose=scene_pose_np,
                        device=str(device),
                        camera=camera,
                        return_depth=False,
                    )
                    render_bgr = cv2.cvtColor(
                        render_rgb_to_u8(np.asarray(result.rgb, dtype=np.float32)),
                        cv2.COLOR_RGB2BGR,
                    )
                    cv2.imwrite(str(output_dir / f"task_{i}_{i + stride}.png"), render_bgr)
                finally:
                    scene_obj.cleanup()

                converged = metrics["converged"]
                euc = float(np.linalg.norm(
                    final_pose[:3].astype(np.float64) - np.asarray(target_pose[:3], dtype=np.float64)
                ))
                R1 = R.from_euler("xyz", final_pose[3:6].astype(np.float64)).as_matrix()
                R2 = R.from_euler("xyz", np.asarray(target_pose[3:6], dtype=np.float64)).as_matrix()
                geo = float(np.arccos(np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)))

                writer.writerow({
                    "step": f"{i}_{i + stride}",
                    "from": frames[i][0].name,
                    "to": target_image.name,
                    "converged": converged,
                    "iterations": metrics["iterations"],
                    "servo_error": round(metrics["final_error"], 6),
                    "euclidean_dist": round(euc, 6),
                    "geodesic_dist_rad": round(geo, 6),
                })
                csv_f.flush()

                viz.add_pose(final_pose, label=f"{i + stride}{'✓' if converged else '✗'}")
                pbar.set_postfix({
                    "step": f"{i}→{i + stride}",
                    "iters": metrics["iterations"],
                    "euc": f"{euc:.4f}",
                    "geo_rad": f"{geo:.4f}",
                    "ok": converged,
                })

                if not converged:
                    break

                all_metrics.append(metrics)
                current_pose = final_pose

    finally:
        viz.save(str(output_dir / "trajectory.jpg"))
        viz.keep_open()

    return all_metrics



__all__ = [
    "fbvs",
    "fbvs_redetect",
    "fbvs_gaussian_redetect",
    "plan_trajectory",
    "_init_scene_and_camera",
    "_init_features_and_world_points",
]
