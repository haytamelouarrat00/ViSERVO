from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from src.camera import create_camera_from_K
from src.control import normalize_features, ZNSSD
from src.features import XFeatMatcher
from src.render_pipeline import render_gaussian_view
from src.servoing.target_io import _load_and_fit_target

# ── Border exclusion matching ViSP's DEFAULT_BORDER ──────────────────────────
DEFAULT_BORDER = 10


# ── Gradient filter ───────────────────────────────────────────────────────────

def _image_gradient_7tap(
    image: np.ndarray, fx: float = 1.0, fy: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    7-tap antisymmetric derivative filter matching ViSP's vpImageFilter:
        dI/du = (2047*(d1) + 913*(d2) + 112*(d3)) / 8418
    Scaled by focal lengths to convert pixel-domain gradient to metric units.
    """
    I = image.astype(np.float64)
    # Kernel: produces (2047*(I[c+1]-I[c-1]) + 913*(I[c+2]-I[c-2]) + 112*(I[c+3]-I[c-3])) / 8418
    kernel = np.array([-112, -913, -2047, 0, 2047, 913, 112], dtype=np.float64) / 8418.0
    Ix = cv2.filter2D(I, cv2.CV_64F, kernel.reshape(1, 7)) * fx
    Iy = cv2.filter2D(I, cv2.CV_64F, kernel.reshape(7, 1)) * fy
    return Ix, Iy


# ── Adaptive gain ─────────────────────────────────────────────────────────────

def _adaptive_gain(
    e1_inf_norm: float,
    gain_at_zero: float,
    gain_at_infinity: float,
    slope_at_zero: float,
) -> float:
    """
    ViSP vpAdaptiveGain:
        λ(x) = (λ₀ − λ∞) · exp(−b · x) + λ∞,   b = slope_at_zero / (λ₀ − λ∞)
    Applied to the L∞ norm of e₁ = L⁺ · (s − s*).
    Higher gain near convergence (small error), lower gain far away.
    """
    a = gain_at_zero - gain_at_infinity
    b = slope_at_zero / (a + 1e-12)
    return a * float(np.exp(-b * e1_inf_norm)) + gain_at_infinity


# ── Luminance Jacobian ────────────────────────────────────────────────────────

def _luminance_jacobian(
    Ix: np.ndarray,
    Iy: np.ndarray,
    depth: np.ndarray | float,
    camera,
    border: int,
) -> np.ndarray:
    """
    Build luminance interaction matrix L_I matching ViSP's
    vpFeatureLuminance::interaction(), shape (N_valid, 6).

    N_valid = (H - 2*border) * (W - 2*border).

    Columns map camera velocity [vx, vy, vz, ωx, ωy, ωz] to pixel intensity rate.
    """
    h, w = Ix.shape

    # Extract valid (border-cropped) region
    Ix_v = Ix[border: h - border, border: w - border].astype(np.float64)
    Iy_v = Iy[border: h - border, border: w - border].astype(np.float64)

    rows_v = h - 2 * border
    cols_v = w - 2 * border

    if np.isscalar(depth):
        Z_v = np.full((rows_v, cols_v), float(depth), dtype=np.float64)
    else:
        Z_v = np.asarray(depth, dtype=np.float64)[border: h - border, border: w - border]

    # Normalised image coordinates for each valid pixel (ViSP: x = (u-u0)/px)
    u_grid, v_grid = np.meshgrid(
        np.arange(border, w - border, dtype=np.float64),
        np.arange(border, h - border, dtype=np.float64),
        indexing="xy",
    )
    pixels = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1).astype(np.float32)
    normalized = normalize_features(pixels, camera).astype(np.float64)
    x = normalized[:, 0].reshape(rows_v, cols_v)
    y = normalized[:, 1].reshape(rows_v, cols_v)

    # Safe inverse depth (fill invalid/zero depths with median)
    valid = np.isfinite(Z_v) & (Z_v > 1e-6)
    fill_z = float(np.median(Z_v[valid])) if np.any(valid) else 1.0
    Z_safe = np.where(valid, Z_v, fill_z)
    inv_z = 1.0 / Z_safe

    # ViSP vpFeatureLuminance::interaction() — exact formula
    L = np.stack([
        Ix_v * inv_z,                                       # dI/dvx
        Iy_v * inv_z,                                       # dI/dvy
        -(x * Ix_v + y * Iy_v) * inv_z,                    # dI/dvz
        -Ix_v * x * y - (1.0 + y * y) * Iy_v,             # dI/dωx
        (1.0 + x * x) * Ix_v + Iy_v * x * y,              # dI/dωy
        Iy_v * x - Ix_v * y,                                # dI/dωz
    ], axis=-1).reshape(-1, 6)

    return L.astype(np.float32)


def _extract_pixels(image: np.ndarray, border: int) -> np.ndarray:
    """Flatten the valid (border-cropped) pixel region as float64."""
    h, w = image.shape[:2]
    return image[border: h - border, border: w - border].astype(np.float64).ravel()


# ── Scene / camera helpers ────────────────────────────────────────────────────

def _infer_camera_size_from_K(K: np.ndarray) -> tuple[int, int]:
    K_arr = np.asarray(K, dtype=np.float64)
    if K_arr.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K_arr.shape}")
    width = int(round(float(K_arr[0, 2]) * 2.0))
    height = int(round(float(K_arr[1, 2]) * 2.0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid inferred camera size: width={width}, height={height}")
    return width, height


def _init_scene_and_camera(scene_path: str | Path, K: np.ndarray, device: torch.device):
    from src.gs import GaussianSplattingScene

    scene_obj = GaussianSplattingScene(device=str(device), verbose=False)
    scene_obj.load_from_ply(str(scene_path))

    if K is None:
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
    from src.control import normalize_features as _nf
    xfeat_matcher = XFeatMatcher(top_k=4096)
    match_result = xfeat_matcher.match(rendered_gray, real_gray)
    kp1, kp2, matches = (
        match_result.keypoints_ref,
        match_result.keypoints_qry,
        match_result.matches,
    )
    tracked_features_px = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    desired_features_px = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    desired_features_norm = _nf(desired_features_px, camera=camera)

    h, w = rendered_depth.shape[:2]
    uv = np.round(tracked_features_px).astype(int)
    uv[:, 0] = np.clip(uv[:, 0], 0, w - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, h - 1)
    init_depths = np.clip(rendered_depth[uv[:, 1], uv[:, 0]], 1e-6, None).astype(np.float32)

    camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
    feature_points_world = camera.back_project_to_world(
        tracked_features_px, init_depths
    ).astype(np.float32)

    return tracked_features_px, desired_features_px, desired_features_norm, feature_points_world


# ── COLMAP pose reader ────────────────────────────────────────────────────────

def get_pose_colmap(
    image_id: int | str | Path,
    path: str | Path = "data/playroom/info/images.txt",
) -> Optional[tuple[np.ndarray, str]]:
    """
    Read a COLMAP image pose and return (pose, image_name).
    pose is [x,y,z,rx,ry,rz] in world frame.
    `image_id` may be an integer ID or a filename / path (matched by basename).
    """
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

        R_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tx, ty, tz], dtype=np.float64)
        R_c2w = R_w2c.T
        C_world = -R_c2w @ t_w2c
        euler_c2w = R.from_matrix(R_c2w).as_euler("xyz")

        pose = np.array(
            [C_world[0], C_world[1], C_world[2],
             euler_c2w[0], euler_c2w[1], euler_c2w[2]],
            dtype=np.float32,
        )
        return pose, image_name

    return None


# ── Main DVS loop ─────────────────────────────────────────────────────────────

def gaussian_dvs(
    scene: str | Path,
    initial_pose: Sequence[float],
    desired_view: str | Path,
    *,
    max_iters: int = 200,
    cost_tolerance: float = 2e3,
    error_tolerance: Optional[float] = None,
    # Adaptive gain parameters (ViSP vpAdaptiveGain defaults)
    gain_at_zero: float = 1.5,
    gain_at_infinity: float = 0.1,
    slope_at_zero: float = 4.0,
    K: np.ndarray = None,
    dt: float = 1.0,
    border: int = DEFAULT_BORDER,
    # Interaction matrix approximation: "current", "desired", or "mean"
    interaction_matrix_type: str = "current",
    verbose: bool = False,
    desired_pose: np.ndarray = None,
):
    """
    Direct Visual Servoing (DVS) following ViSP's vpFeatureLuminance approach.
    Servos a Gaussian-Splatting render to match a real target image.

    Control law  (ViSP Eq.):
        v = −λ(‖L⁺e‖∞) · L⁺ · (s − s*)

    where
        s       = raw 0-255 pixel intensities of the current GS render
        s*      = raw 0-255 pixel intensities of the real target image
        L       = luminance interaction matrix (7-tap gradients, per-pixel depth)
        λ(·)    = adaptive gain (high near convergence, lower far away)

    Args:
        interaction_matrix_type:
            "current"  — L re-evaluated from current render each iteration (default)
            "desired"  — L computed once from GS render at desired_pose, then fixed
            "mean"     — average of L_current and L_desired each iteration
    """
    camera_pose = np.asarray(initial_pose, dtype=np.float32).copy()
    if camera_pose.size != 6:
        raise ValueError(f"initial_pose must have 6 elements, got {camera_pose.shape}")

    scene_pose_np = np.zeros(6, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_obj, camera, cam_w, cam_h = _init_scene_and_camera(scene, K, device)

    fx = float(camera.intrinsics.fx)
    fy = float(camera.intrinsics.fy)

    # Load target image — raw uint8, convert to float32 (0-255), no normalization
    real_bgr, real_gray_raw = _load_and_fit_target(desired_view, cam_w, cam_h)
    real_gray_f = real_gray_raw.astype(np.float32)   # 0-255 floats
    s_desired = _extract_pixels(real_gray_f, border)  # flat float64 pixel vector

    # ── Pre-compute L_desired for "desired" / "mean" modes ───────────────────
    L_desired = None
    if interaction_matrix_type in ("desired", "mean"):
        if desired_pose is not None:
            des_pose_np = np.asarray(desired_pose, dtype=np.float32)
            camera.set_pose(position=des_pose_np[:3], orientation=des_pose_np[3:])
            res_des = render_gaussian_view(
                scene=scene_obj, camera_pose=des_pose_np,
                scene_pose=scene_pose_np, device=str(device),
                camera=camera, return_depth=True,
            )
            des_rgb = np.asarray(res_des.rgb, dtype=np.float32)
            des_gray_f = np.clip(
                cv2.cvtColor(des_rgb, cv2.COLOR_RGB2GRAY) * 255.0, 0, 255
            ).astype(np.float32)
            des_depth = np.asarray(res_des.depth, dtype=np.float32)
            Ix_d, Iy_d = _image_gradient_7tap(des_gray_f, fx=fx, fy=fy)
            L_desired = _luminance_jacobian(Ix_d, Iy_d, des_depth, camera, border)
        else:
            # Fallback: gradients from real photo, constant depth Z=1
            Ix_d, Iy_d = _image_gradient_7tap(real_gray_f, fx=fx, fy=fy)
            L_desired = _luminance_jacobian(Ix_d, Iy_d, 1.0, camera, border)

    converged = False
    iteration = 0
    cost_history: list[float] = []
    last_znssd = float("nan")

    try:
        while not converged and iteration < max_iters:

            # 1. Render current view — convert to raw 0-255 float (no normalization)
            camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])
            result = render_gaussian_view(
                scene=scene_obj, camera_pose=camera_pose,
                scene_pose=scene_pose_np, device=str(device),
                camera=camera, return_depth=True,
            )
            render_rgb = np.asarray(result.rgb, dtype=np.float32)
            render_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
            render_gray_f = np.clip(render_gray * 255.0, 0, 255).astype(np.float32)
            render_depth = np.asarray(result.depth, dtype=np.float32)

            # 2. Feature error: e = s − s*  (raw pixel difference, no ZN)
            s_current = _extract_pixels(render_gray_f, border)
            error = s_current - s_desired
            cost = 0.5 * float(np.dot(error, error))
            cost_history.append(cost)

            # 3. Interaction matrix
            Ix, Iy = _image_gradient_7tap(render_gray_f, fx=fx, fy=fy)
            L_current = _luminance_jacobian(Ix, Iy, render_depth, camera, border)

            if interaction_matrix_type == "desired" and L_desired is not None:
                L = L_desired
            elif interaction_matrix_type == "mean" and L_desired is not None:
                L = 0.5 * (L_current + L_desired)
            else:
                L = L_current   # CURRENT (default)

            # 4. Solve: e₁ = L⁺ · e  via normal equations (6×6 system)
            L_64 = L.astype(np.float64)
            H = L_64.T @ L_64                                    # (6,6)
            g = L_64.T @ error                                   # (6,)
            damp = 1e-6 * (float(np.trace(H)) / 6.0 + 1e-12)
            e1 = np.linalg.solve(H + damp * np.eye(6), g)       # L⁺ · e

            # 5. Adaptive gain on L∞ norm of e₁ (ViSP vpAdaptiveGain)
            lam = _adaptive_gain(
                float(np.max(np.abs(e1))),
                gain_at_zero, gain_at_infinity, slope_at_zero,
            )

            # 6. Control law: v = −λ · L⁺ · e
            v_cmd = (-lam * e1).astype(np.float32)

            # Clamp to keep camera inside scene bounds
            t_norm = float(np.linalg.norm(v_cmd[:3]))
            r_norm = float(np.linalg.norm(v_cmd[3:]))
            MAX_T, MAX_R = 0.3, 0.3
            if t_norm > MAX_T:
                v_cmd[:3] *= MAX_T / t_norm
            if r_norm > MAX_R:
                v_cmd[3:] *= MAX_R / r_norm

            znssd = ZNSSD(real_gray_f, render_gray_f)
            last_znssd = znssd

            # 8. Convergence
            if cost < cost_tolerance:
                converged = True
                break
            if error_tolerance is not None and znssd < error_tolerance:
                converged = True
                break

            # 9. Integrate velocity
            camera.apply_velocity(v_cmd, dt=dt, frame="camera")
            position, orientation = camera.get_pose()
            camera_pose = np.concatenate([position, orientation]).astype(np.float32)
            iteration += 1

    finally:
        scene_obj.cleanup()

    final_cost = cost_history[-1] if cost_history else float("nan")

    return {
        "converged": bool(converged),
        "iterations": int(iteration),
        "final_pose": camera_pose.copy(),
        "final_cost": final_cost,
        "final_error": last_znssd,
    }
