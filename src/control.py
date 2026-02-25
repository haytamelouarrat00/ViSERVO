from src.camera import VirtualCamera
import numpy as np
import cv2

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

def _to_intensity_float255(image: np.ndarray) -> np.ndarray:
    """Convert image intensities to float64 in the [0, 255] domain when possible."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return arr

    max_val = float(np.max(arr))
    min_val = float(np.min(arr))
    if min_val >= 0.0 and max_val <= 1.0 + 1e-6:
        return arr * 255.0
    return arr


def photometric_error(
    desired_image: np.ndarray, current_image: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute direct photometric error e = I(r) - I(r*), flattened.

    Returns:
        error: Flattened error vector of shape (H*W,).
        cost: Squared L2 cost e^T e.
    """
    desired = _to_intensity_float255(desired_image)
    current = _to_intensity_float255(current_image)
    if desired.shape != current.shape:
        raise ValueError(
            f"Shape mismatch: desired_image has shape {desired.shape}, "
            f"but current_image has shape {current.shape}"
        )

    error = (current - desired).reshape(-1)
    cost = 1/2 * float(error @ error)
    return error, cost


def luminance_interaction_matrix(
    Ix: np.ndarray,
    Iy: np.ndarray,
    depth: np.ndarray | float,
    camera: VirtualCamera,
) -> np.ndarray:
    """
    Compute dense luminance interaction matrix L_I (Eq. 20), shape (H*W, 6).
    """
    Ix_arr = np.asarray(Ix, dtype=np.float64)
    Iy_arr = np.asarray(Iy, dtype=np.float64)
    if Ix_arr.shape != Iy_arr.shape:
        raise ValueError(f"Ix and Iy must have same shape, got {Ix_arr.shape} vs {Iy_arr.shape}")
    if Ix_arr.ndim != 2:
        raise ValueError(f"Ix and Iy must be 2D, got {Ix_arr.shape}")

    h, w = Ix_arr.shape
    if np.isscalar(depth):
        depth_arr = np.full((h, w), float(depth), dtype=np.float64)
    else:
        depth_arr = np.asarray(depth, dtype=np.float64)
        if depth_arr.shape != (h, w):
            raise ValueError(f"Depth shape must match image shape {(h, w)}, got {depth_arr.shape}")

    valid = np.isfinite(depth_arr) & (depth_arr > 1e-6)
    if not np.any(valid):
        raise ValueError("Depth map has no finite positive values.")
    fill_depth = float(np.median(depth_arr[valid]))
    depth_safe = np.where(valid, depth_arr, fill_depth)
    inv_z = 1.0 / depth_safe

    u, v = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64),
        indexing="xy",
    )
    pixels = np.stack([u.reshape(-1), v.reshape(-1)], axis=1).astype(np.float32)
    normalized = normalize_features(pixels, camera).astype(np.float64)
    x = normalized[:, 0].reshape(h, w)
    y = normalized[:, 1].reshape(h, w)

    Lx = np.stack(
        [-inv_z, np.zeros_like(x), x * inv_z, x * y, -(1.0 + x * x), y],
        axis=-1,
    )
    Ly = np.stack(
        [np.zeros_like(y), -inv_z, y * inv_z, 1.0 + y * y, -x * y, -x],
        axis=-1,
    )

    L_I = -(Ix_arr[..., np.newaxis] * Lx + Iy_arr[..., np.newaxis] * Ly)
    return L_I.reshape(-1, 6).astype(np.float32)


def visualize_photometric_error(
    I_current: np.ndarray, I_desired: np.ndarray, vmin=None, vmax=None
) -> np.ndarray:
    """Visualize direct photometric residual map e = I(current) - I(desired)."""
    import matplotlib.pyplot as plt

    if I_current.ndim == 3:
        I_current = cv2.cvtColor(I_current, cv2.COLOR_BGR2GRAY)
    if I_desired.ndim == 3:
        I_desired = cv2.cvtColor(I_desired, cv2.COLOR_BGR2GRAY)

    I_curr = _to_intensity_float255(I_current)
    I_des = _to_intensity_float255(I_desired)
    if I_curr.shape != I_des.shape:
        raise ValueError(f"Image shapes must match: {I_curr.shape} vs {I_des.shape}")

    error = I_curr - I_des
    if vmin is None or vmax is None:
        scale = float(np.percentile(np.abs(error), 99.0))
        scale = max(scale, 1.0)
        if vmin is None:
            vmin = -scale
        if vmax is None:
            vmax = scale

    plt.figure(figsize=(6, 5))
    plt.imshow(error, cmap="seismic", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Intensity Difference (I - I*)")
    plt.title("Photometric Error: I(current) - I(desired)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return error


def image_gradient(
    image: np.ndarray, fx: float = 1.0, fy: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute image gradients using OpenCV's Scharr operator (more accurate than Sobel)"""

    # Convert to float for accuracy
    I = image.astype(np.float64)

    # Scharr operator (better rotational symmetry than Sobel)
    Ix = cv2.Scharr(I, cv2.CV_64F, 1, 0) * fx  # dx=1, dy=0
    Iy = cv2.Scharr(I, cv2.CV_64F, 0, 1) * fy  # dx=0, dy=1

    return Ix, Iy


import numpy as np
from typing import Tuple, Optional


class PhotometricVisualServoingZN:
    def __init__(
            self,
            camera: VirtualCamera,
            lambda_gain: float = 1.0,
            use_zero_mean_normalization: bool = True,
            use_gradient_magnitude: bool = False,
            verbose: bool = False,
    ):
        """
        Zero-mean Normalized Photometric Visual Servoing
        Based on Rodriguez et al. (2020) - ICRA

        Args:
            camera: VirtualCamera object with get_K()
            lambda_gain: Proportional gain λ (Eq. 3 from Rodriguez paper)
            use_zero_mean_normalization: Apply ZN for lighting robustness
            use_gradient_magnitude: Use gradient magnitude instead of intensity
            verbose: Print debug information
        """
        self.camera = camera
        self.lambda_gain = float(lambda_gain)
        self.use_zn = bool(use_zero_mean_normalization)
        self.use_grad_mag = bool(use_gradient_magnitude)
        self.verbose = bool(verbose)

        # State
        self.cost_history = []
        self.converged = False

    def zero_mean_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Zero-mean Normalization (ZN) to image.
        Makes the metric robust to affine lighting changes (α*I + β).

        Args:
            image: Input image (H, W) or (H, W, 1)

        Returns:
            Zero-mean normalized image (H, W)
        """
        I = image.astype(np.float64)
        if I.ndim == 3:
            I = I.squeeze()

        # Zero-mean normalization (Eq. 2 from Rodriguez paper)
        mean = np.mean(I)
        I_normalized = I - mean

        # Optional: Full ZNCC (divide by std) - Rodriguez uses zero-mean only
        # std = np.std(I)
        # if std > 1e-8:
        #     I_normalized = I_normalized / std

        return I_normalized

    def compute_photometric_error(
            self,
            I_current: np.ndarray,
            I_desired: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute photometric error with Zero-mean Normalization.
        e = Ĩ(r) - Ĩ(r*)  (Eq. 2 from Rodriguez paper)

        Args:
            I_current: Current image (H, W)
            I_desired: Desired image (H, W)

        Returns:
            error: Flattened error vector (N,)
            cost: Cost function value C(ξ) = 0.5 * ||e||²
        """
        # Apply Zero-mean Normalization if enabled
        if self.use_zn:
            I_curr_norm = self.zero_mean_normalize(I_current)
            I_des_norm = self.zero_mean_normalize(I_desired)
        else:
            I_curr_norm = I_current.astype(np.float64)
            I_des_norm = I_desired.astype(np.float64)

        # Compute error vector
        error = (I_curr_norm - I_des_norm).flatten()

        # Compute cost function C(ξ) = 0.5 * ||e||² (Eq. 2)
        cost = 0.5 * float(np.sum(error ** 2))

        return error, cost

    def compute_gradient_magnitude_error(
            self,
            Ix_curr: np.ndarray,
            Iy_curr: np.ndarray,
            Ix_des: np.ndarray,
            Iy_des: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Use gradient magnitude instead of intensity."""
        Ix_curr = np.asarray(Ix_curr, dtype=np.float64)
        Iy_curr = np.asarray(Iy_curr, dtype=np.float64)
        Ix_des = np.asarray(Ix_des, dtype=np.float64)
        Iy_des = np.asarray(Iy_des, dtype=np.float64)

        if Ix_curr.shape != Iy_curr.shape:
            raise ValueError(
                f"Current gradient shapes must match: {Ix_curr.shape} vs {Iy_curr.shape}"
            )
        if Ix_des.shape != Iy_des.shape:
            raise ValueError(
                f"Desired gradient shapes must match: {Ix_des.shape} vs {Iy_des.shape}"
            )
        if Ix_curr.shape != Ix_des.shape:
            raise ValueError(
                f"Current and desired gradient shapes must match: {Ix_curr.shape} vs {Ix_des.shape}"
            )

        mag_curr = np.sqrt(Ix_curr ** 2 + Iy_curr ** 2)
        mag_des = np.sqrt(Ix_des ** 2 + Iy_des ** 2)

        # Apply ZN to gradient magnitudes
        if self.use_zn:
            mag_curr = self.zero_mean_normalize(mag_curr)
            mag_des = self.zero_mean_normalize(mag_des)

        error = (mag_curr - mag_des).flatten()
        cost = 0.5 * float(np.sum(error ** 2))

        return error, cost

    def compute_interaction_matrix(
            self,
            Ix: np.ndarray,
            Iy: np.ndarray,
            Z: float | np.ndarray,
            camera: VirtualCamera
    ) -> np.ndarray:
        """
        Compute luminance interaction matrix L_Ĩ.
        L_Ĩ = -∇Ĩᵀ * L_x  (Eq. 4 from Rodriguez paper)

        Args:
            Ix, Iy: Spatial image gradients (H, W), already scaled by fx, fy
            Z: Depth (scalar or per-pixel map)
            camera: VirtualCamera object

        Returns:
            L_I: Interaction matrix (N, 6) where N = H*W
        """
        return luminance_interaction_matrix(Ix=Ix, Iy=Iy, depth=Z, camera=camera)

    def compute_control_velocity(
            self,
            I_current: np.ndarray,
            I_desired: np.ndarray,
            Ix: np.ndarray,
            Iy: np.ndarray,
            Z: float | np.ndarray,
            Ix_des: Optional[np.ndarray] = None,
            Iy_des: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute camera velocity using Gauss-Newton control law.
        v = -λ * L_Ĩ⁺ * (Ĩ(ξ) - Ĩ(ξ*))  (Eq. 3 from Rodriguez paper)

        Args:
            I_current: Current image (H, W)
            I_desired: Desired image (H, W)
            Ix, Iy: Spatial gradients of current image (H, W)
            Z: Depth estimate (scalar or per-pixel map)
            Ix_des, Iy_des: Optional desired-image gradients (H, W)

        Returns:
            v: Camera velocity (6,) [vx, vy, vz, ωx, ωy, ωz]
            cost: Current cost function value
        """
        # 1. Compute residual vector and cost.
        if self.use_grad_mag:
            if Ix_des is None or Iy_des is None:
                fx = float(self.camera.intrinsics.fx)
                fy = float(self.camera.intrinsics.fy)
                Ix_des, Iy_des = image_gradient(I_desired, fx=fx, fy=fy)
            error, cost = self.compute_gradient_magnitude_error(
                Ix_curr=Ix,
                Iy_curr=Iy,
                Ix_des=Ix_des,
                Iy_des=Iy_des,
            )
        else:
            error, cost = self.compute_photometric_error(I_current, I_desired)
        self.cost_history.append(cost)

        # 2. Compute interaction matrix.
        L_I = self.compute_interaction_matrix(Ix, Iy, Z, self.camera)

        # 3. Gauss-Newton control law (Eq. 3)
        # v = -λ * L_I⁺ * e
        try:
            # Moore-Penrose pseudo-inverse
            L_I_pinv = np.linalg.pinv(L_I)
            v = -self.lambda_gain * L_I_pinv @ error
        except np.linalg.LinAlgError:
            # Fallback: damped least squares
            H = L_I.T @ L_I
            H_damped = H + 1e-6 * np.eye(6)
            v = -self.lambda_gain * np.linalg.solve(H_damped, L_I.T @ error)

        # 4. Check convergence
        velocity_norm = float(np.linalg.norm(v))
        if velocity_norm < 1e-6 and len(self.cost_history) > 10:
            self.converged = True

        if self.verbose and len(self.cost_history) % 10 == 0:
            print(f"[dvs_zn] Iter {len(self.cost_history)}: "
                  f"cost={cost:.2e}, ||v||={velocity_norm:.4f}, "
                  f"converged={self.converged}")

        return np.asarray(v, dtype=np.float32).reshape(6), float(cost)

    def reset(self):
        """Reset state for new servoing task"""
        self.cost_history = []
        self.converged = False

    def get_convergence_stats(self) -> dict:
        """Get statistics about convergence"""
        if len(self.cost_history) == 0:
            return {}

        return {
            'iterations': len(self.cost_history),
            'initial_cost': self.cost_history[0],
            'final_cost': self.cost_history[-1],
            'cost_reduction': self.cost_history[0] / (self.cost_history[-1] + 1e-10),
            'converged': self.converged,
        }


# Backward compatibility for existing call sites.
PhotometricVisualServoingMLM = PhotometricVisualServoingZN
import numpy as np
import cv2


def normalize_photometric_affine(I_real, I_rendered):
    """
    Normalize real image to match rendered image statistics.
    I_norm = α * I_real + β
    """
    # Compute statistics
    μ_real = np.mean(I_real)
    σ_real = np.std(I_real)

    μ_render = np.mean(I_rendered)
    σ_render = np.std(I_rendered)

    # Compute affine parameters
    α = σ_render / (σ_real + 1e-8)
    β = μ_render - α * μ_real

    # Apply normalization
    I_normalized = α * I_real.astype(np.float64) + β

    # Clip to valid range
    I_normalized = np.clip(I_normalized, 0, 255).astype(np.float64)

    return I_normalized
