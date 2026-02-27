"""Backward-compatible FBVS/DVS public API.

This module preserves legacy import paths (`from src.fbvs import ...`) while the
implementation is organized under `src.servoing`.
"""

from src.servoing import (
    _compose_current_target_view,
    _draw_iteration_features,
    _load_and_fit_target,
    _make_frame_output_dir,
    _make_video_writer,
    _safe_imshow,
    dvs_mesh,
    fbvs,
    fbvs_gaussian_redetect,
    fbvs_redetect,
    fbvs_mesh,
    get_features_depth,
    render_view,
)
from src.servoing.gaussian_fbvs import _init_features_and_world_points, _init_scene_and_camera

__all__ = [
    "fbvs",
    "fbvs_redetect",
    "fbvs_gaussian_redetect",
    "fbvs_mesh",
    "dvs_mesh",
    "render_view",
    "get_features_depth",
    "_compose_current_target_view",
    "_draw_iteration_features",
    "_safe_imshow",
    "_init_scene_and_camera",
    "_load_and_fit_target",
    "_init_features_and_world_points",
    "_make_video_writer",
    "_make_frame_output_dir",
]
