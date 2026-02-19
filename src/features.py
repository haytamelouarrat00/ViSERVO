from __future__ import annotations

import contextlib
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ImageArray = NDArray[np.uint8]
KeyPoints = NDArray[np.object_]
Matches = NDArray[np.object_]


# ---------------------------------------------------------------------------
# Structured return type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatchResult:
    keypoints_ref: KeyPoints
    keypoints_qry: KeyPoints
    matches: Matches


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _validate_grayscale_image(image: np.ndarray, name: str = "image") -> None:
    if image.ndim != 2:
        raise ValueError(f"{name} must be a 2-D grayscale array, got shape {image.shape}.")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if image.dtype not in (np.uint8, np.float32, np.float64):
        raise ValueError(
            f"{name} has unsupported dtype {image.dtype}. "
            "Expected uint8, float32, or float64."
        )


def _to_uint8(image: np.ndarray) -> ImageArray:
    if image.dtype == np.uint8:
        return image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _resize_to_match(source: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(source, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Matcher interface
# ---------------------------------------------------------------------------
class FeatureMatcher(ABC):
    @abstractmethod
    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        ...


# ---------------------------------------------------------------------------
# SIFT matcher
# ---------------------------------------------------------------------------
class SIFTMatcher(FeatureMatcher):
    def __init__(
        self,
        ratio_threshold: float = 0.7,
        flann_trees: int = 5,
        flann_checks: int = 50,
    ) -> None:
        if not (0.0 < ratio_threshold < 1.0):
            raise ValueError(f"ratio_threshold must be in (0, 1), got {ratio_threshold}.")

        self._ratio_threshold = ratio_threshold
        self._sift = cv2.SIFT_create()
        self._flann = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": flann_trees},
            {"checks": flann_checks},
        )

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        ref_u8 = _to_uint8(reference)
        qry_u8 = _to_uint8(query)

        kp1, des1 = self._sift.detectAndCompute(ref_u8, None)
        kp2, des2 = self._sift.detectAndCompute(qry_u8, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            logger.warning(
                "SIFT: insufficient features detected (ref=%d, qry=%d).",
                len(kp1),
                len(kp2),
            )
            return MatchResult(
                np.array(kp1, dtype=object),
                np.array(kp2, dtype=object),
                np.empty((0,), dtype=object),
            )

        raw_matches = self._flann.knnMatch(des1, des2, k=2)
        good = [
            m
            for pair in raw_matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < self._ratio_threshold * n.distance
        ]

        return MatchResult(
            np.array(kp1, dtype=object),
            np.array(kp2, dtype=object),
            np.array(good, dtype=object),
        )


# ---------------------------------------------------------------------------
# XFeat matcher
# ---------------------------------------------------------------------------
class XFeatMatcher(FeatureMatcher):
    def __init__(self, top_k: int = 4096, xfeat_root: str | None = None) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        self._top_k = top_k
        self._xfeat_root = xfeat_root
        self._xfeat: object | None = None

    def _get_xfeat(self) -> object:
        if self._xfeat is not None:
            return self._xfeat

        import sys
        from pathlib import Path

        if self._xfeat_root is not None:
            if self._xfeat_root not in sys.path:
                sys.path.insert(0, self._xfeat_root)
        else:
            default_root = str(Path(__file__).resolve().parents[1] / "accelerated_features")
            if Path(default_root).exists() and default_root not in sys.path:
                sys.path.insert(0, default_root)

        from accelerated_features.modules.xfeat import XFeat  # type: ignore

        # Silence third-party constructor prints (e.g. weight-path banner).
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._xfeat = XFeat()
        return self._xfeat

    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult:
        xfeat = self._get_xfeat()

        ref_u8 = _to_uint8(reference)
        qry_u8 = _to_uint8(query)
        ref_bgr = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
        qry_bgr = cv2.cvtColor(qry_u8, cv2.COLOR_GRAY2BGR)

        mkpts_0, mkpts_1 = xfeat.match_xfeat(ref_bgr, qry_bgr, top_k=self._top_k)
        mkpts_0 = np.asarray(mkpts_0, dtype=np.float32).reshape(-1, 2)
        mkpts_1 = np.asarray(mkpts_1, dtype=np.float32).reshape(-1, 2)

        n = min(len(mkpts_0), len(mkpts_1))
        mkpts_0 = mkpts_0[:n]
        mkpts_1 = mkpts_1[:n]

        kp1 = np.array([cv2.KeyPoint(float(p[0]), float(p[1]), 5.0) for p in mkpts_0], dtype=object)
        kp2 = np.array([cv2.KeyPoint(float(p[0]), float(p[1]), 5.0) for p in mkpts_1], dtype=object)

        if n < 4:
            matches = np.array([cv2.DMatch(i, i, 0.0) for i in range(n)], dtype=object)
            return MatchResult(kp1, kp2, matches)

        H, mask = cv2.findHomography(
            mkpts_0,
            mkpts_1,
            cv2.USAC_MAGSAC,
            3.5,
            maxIters=1_000,
            confidence=0.999,
        )

        if H is None or mask is None:
            matches = np.array([cv2.DMatch(i, i, 0.0) for i in range(n)], dtype=object)
            return MatchResult(kp1, kp2, matches)

        inlier_mask = mask.ravel().astype(bool)
        matches = np.array(
            [cv2.DMatch(i, i, 0.0) for i in range(len(inlier_mask)) if inlier_mask[i]],
            dtype=object,
        )

        return MatchResult(kp1, kp2, matches)


# ---------------------------------------------------------------------------
# Thin compatibility facade
# ---------------------------------------------------------------------------
class Features:
    def __init__(self, reference: np.ndarray, query: np.ndarray) -> None:
        _validate_grayscale_image(reference, "reference")
        _validate_grayscale_image(query, "query")

        self.reference = reference
        self.query = (
            _resize_to_match(query, reference.shape[:2])
            if query.shape[:2] != reference.shape[:2]
            else query
        )

    def match_sift(self, ratio_threshold: float = 0.7):
        result = SIFTMatcher(ratio_threshold=ratio_threshold).match(
            self.reference, self.query
        )
        return result.keypoints_ref, result.keypoints_qry, result.matches

    def match_xfeat(self, top_k: int = 4096, xfeat_root: str | None = None):
        result = XFeatMatcher(top_k=top_k, xfeat_root=xfeat_root).match(
            self.reference, self.query
        )
        return result.keypoints_ref, result.keypoints_qry, result.matches
