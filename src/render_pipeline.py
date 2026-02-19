from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.camera import create_default_camera
from src.gs import GaussianParameters, GaussianSplattingScene
from src.scene_ops import add_world_frame_gaussians, apply_scene_pose, scene_bounds


@dataclass
class RenderResult:
    rgb: np.ndarray
    alpha: Optional[np.ndarray]
    params: Optional[GaussianParameters] = None
    depth: Optional[np.ndarray] = None


def render_gaussian_view(
    ply_path: str | Path | None = None,
    camera_pose: np.ndarray = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
    *,
    scene_pose: np.ndarray = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
    device: str = "cuda",
    width: int = 1920,
    height: int = 1080,
    fov_deg: float = 60.0,
    near: float = 0.01,
    far: float = 100.0,
    include_world_frame: bool = False,
    axis_length: float | None = None,
    axis_samples: int = 64,
    axis_scale: float = 0.01,
    return_alpha: bool = False,
    return_depth: bool = False,
    depth_mode: str = "ED",
    scene_verbose: bool = False,
    camera=None,
    scene: GaussianSplattingScene | None = None,
) -> RenderResult:
    """
    Shared rendering pipeline used by CLI and debugging tools.
    """
    camera_pose_np = np.asarray(camera_pose, dtype=np.float32)
    scene_pose_np = np.asarray(scene_pose, dtype=np.float32)
    if camera_pose_np.shape != (6,):
        raise ValueError(
            f"camera_pose must have 6 values [x,y,z,rx,ry,rz], got {camera_pose_np.shape}"
        )
    if scene_pose_np.shape != (6,):
        raise ValueError(
            f"scene_pose must have 6 values [x,y,z,rx,ry,rz], got {scene_pose_np.shape}"
        )

    params = None
    if scene is None:
        if ply_path is None:
            raise ValueError("Either 'ply_path' or 'scene' must be provided.")
        scene = GaussianSplattingScene(device=device, verbose=scene_verbose)
        params = scene.load_from_ply(str(ply_path))
        params = apply_scene_pose(params, scene_pose_np)
        scene.load_gaussians(params)

        if include_world_frame:
            _, _, _, radius = scene_bounds(params.means)
            axis_length_value = axis_length if axis_length is not None else 0.6 * radius
            params = add_world_frame_gaussians(
                params=params,
                axis_length=float(max(axis_length_value, 1e-3)),
                samples_per_axis=max(axis_samples, 2),
                axis_scale=float(max(axis_scale, 1e-5)),
            )
            scene.load_gaussians(params)
    else:
        # If scene is provided, we assume it's already loaded with desired params.
        pass

    if camera is None:
        camera = create_default_camera(width=width, height=height, fov_deg=fov_deg)

    camera.set_pose(position=camera_pose_np[:3])
    camera.set_pose(orientation=camera_pose_np[3:])

    try:
        rendered = scene.render_from_virtual_camera(
            camera, near=near, far=far, return_alpha=return_alpha
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Gaussian rendering failed. gsplat in this environment likely requires a CUDA "
            "device. Run with `device='cuda'` on a CUDA-capable machine."
        ) from exc

    if isinstance(rendered, tuple):
        rgb, alpha = rendered
    else:
        rgb, alpha = rendered, None

    depth = None
    if return_depth:
        depth = scene.render_depth_from_virtual_camera(
            camera,
            near=near,
            far=far,
            depth_mode=depth_mode,
            return_alpha=False,
        )

    return RenderResult(rgb=rgb, alpha=alpha, params=params, depth=depth)
