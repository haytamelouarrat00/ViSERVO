from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


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


def render_rgb_to_u8(render_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB render to uint8 while preserving existing conversion rules."""
    if np.issubdtype(render_rgb.dtype, np.floating):
        if float(np.max(render_rgb)) <= 1.0:
            return np.clip(render_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.clip(render_rgb, 0.0, 255.0).astype(np.uint8)
    if render_rgb.dtype == np.uint8:
        return render_rgb
    return np.clip(render_rgb, 0.0, 255.0).astype(np.uint8)


__all__ = [
    "_compose_current_target_view",
    "_draw_iteration_features",
    "_safe_imshow",
    "_make_video_writer",
    "_make_frame_output_dir",
    "render_rgb_to_u8",
]
