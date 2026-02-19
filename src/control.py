import numpy as np

from src.features import Features
from src.camera import VirtualCamera


def geometric_error(
    desired_features: np.ndarray, current_features: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute the geometric error between the desired and current features.
    Returns:
        errors: An array of shape (N,) containing the per-feature errors.
        mean_error: The mean error across all features.
    """
    if desired_features.shape != current_features.shape:
        raise ValueError(
            f"Shape mismatch: desired_features has shape {desired_features.shape}, "
            f"but current_features has shape {current_features.shape}"
        )
    desired_features = desired_features.reshape(-1, 1)
    current_features = current_features.reshape(-1, 1)
    error = current_features - desired_features

    # Compute per-feature errors (L2 distance)
    error_norm = np.linalg.norm(error)

    return error, float(error_norm)  # in pixels


def ZNSSD(
    desired_features: np.ndarray, current_features: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute the Zero-mean Normalized Sum of Squared Differences (ZNSSD) error between the desired and current features.
    Returns:
        errors: An array of shape (N,) containing the per-feature ZNSSD errors.
        mean_error: The mean ZNSSD error across all features.
    """
    if desired_features.shape != current_features.shape:
        raise ValueError(
            f"Shape mismatch: desired_features has shape {desired_features.shape}, "
            f"but current_features has shape {current_features.shape}"
        )
    desired_mean = np.mean(desired_features)
    current_mean = np.mean(current_features)

    desired_zero_mean = desired_features - desired_mean
    current_zero_mean = current_features - current_mean

    numerator = np.sum((current_zero_mean - desired_zero_mean) ** 2)
    denominator = np.sqrt(np.sum(desired_zero_mean**2) * np.sum(current_zero_mean**2))

    if denominator == 0:
        raise ValueError("Denominator in ZNSSD calculation is zero.")

    znssd_error = numerator / denominator

    return znssd_error


def normalize_features(features: np.ndarray, camera: VirtualCamera) -> np.ndarray:
    """
    Normalize pixel coordinates to normalized image coordinates using the camera intrinsics.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(
            f"Expected features to have shape (N, 2), got {features.shape}"
        )

    K = camera.get_K()
    K_inv = np.linalg.inv(K)

    # Convert [u, v] to homogeneous [u, v, 1], apply K^-1, and keep [x, y].
    ones = np.ones((features.shape[0], 1), dtype=np.float32)
    features_h = np.concatenate([features, ones], axis=1)  # (N, 3)
    normalized_h = (K_inv @ features_h.T).T  # (N, 3)
    return normalized_h[:, :2].astype(np.float32)


def interaction_matrix(features: np.ndarray, depths) -> np.ndarray:
    """
    Compute the IBVS interaction matrix (image Jacobian) for point features.

    Args:
        features: Normalized image coordinates with shape (N, 2), each row [x, y].
        depths: Either a scalar depth or per-point depths with shape (N,) / (N, 1).

    Returns:
        Interaction matrix with shape (2N, 6), mapping camera twist
        [vx, vy, vz, wx, wy, wz] to image feature velocity.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(
            f"Expected features to have shape (N, 2), got {features.shape}"
        )

    n = features.shape[0]
    depths_arr = np.asarray(depths, dtype=np.float64).reshape(-1)
    if depths_arr.size == 1:
        depths_arr = np.full(n, float(depths_arr[0]), dtype=np.float64)
    elif depths_arr.size != n:
        raise ValueError(
            f"Depth size mismatch: expected 1 or {n} values, got {depths_arr.size}"
        )

    if np.any(depths_arr <= 0):
        raise ValueError("All depth values must be positive.")

    L = np.zeros((2 * n, 6), dtype=np.float32)
    for i, ((x, y), z) in enumerate(zip(features, depths_arr)):
        L[2 * i] = [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y]
        L[2 * i + 1] = [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x]

    return L


def velocity(L: np.ndarray, error: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """
    Compute the camera velocity command given the interaction matrix and feature error.

    Args:
        L: Interaction matrix of shape (2N, 6).
        error: Feature error vector of shape (2N,).
        gain: Proportional gain for the controller.
    Returns:
        Camera velocity command as a vector of shape (6,).
    """
    L = np.asarray(L, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64).reshape(-1)

    if L.ndim != 2:
        raise ValueError(f"Expected L to be a 2D matrix, got shape {L.shape}")
    if L.shape[0] != error.shape[0]:
        raise ValueError(
            f"Row mismatch: L has {L.shape[0]} rows but error has {error.shape[0]} elements"
        )
    if L.shape[1] != 6:
        raise ValueError(f"Expected L to have 6 columns, got {L.shape[1]}")

    # Compute the pseudo-inverse of L
    L_pinv = np.linalg.pinv(L)
    v = -float(gain) * (L_pinv @ error)
    return np.asarray(v, dtype=np.float32).reshape(6)
