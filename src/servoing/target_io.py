from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _load_and_fit_target(
    desired_view_path: str | Path, cam_w: int, cam_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load desired view and resize to camera resolution; return (bgr, gray)."""
    real_bgr = cv2.imread(str(desired_view_path), cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise FileNotFoundError(f"Could not read image: {desired_view_path}")

    if real_bgr.shape[:2] != (cam_h, cam_w):
        real_bgr = cv2.resize(real_bgr, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)

    return real_bgr, cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)


__all__ = ["_load_and_fit_target"]
