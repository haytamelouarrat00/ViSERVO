"""
Virtual Camera Class for Open3D
A minimalistic implementation for visual servoing and 3D scene simulation.

Author: Claude
Date: February 2026
"""

import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
from scipy.spatial.transform import Rotation as R
from typing import Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy.linalg import expm

def _as_vector(x: np.ndarray, size: int, name: str) -> np.ndarray:
    """Convert input to a flat vector with a fixed number of elements."""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size != size:
        raise ValueError(f"{name} must have {size} elements, got shape {x_arr.shape}")
    return x_arr.reshape(size)


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Compute the SE(3) exponential map: exp(ξ^).

    Converts a 6D twist (element of se(3) Lie algebra) to a 4x4
    homogeneous transformation matrix (element of SE(3) Lie group).

    Args:
        xi: 6D twist vector [vx, vy, vz, wx, wy, wz]

    Returns:
        4x4 homogeneous transformation matrix in SE(3)

    Reference:
        "A Mathematical Introduction to Robotic Manipulation"
        Murray, Li, Sastry (1994), Chapter 2
    """
    xi = _as_vector(xi, size=6, name="xi")
    v = xi[:3]  # Linear velocity
    w = xi[3:]  # Angular velocity (rotation vector)

    theta = np.linalg.norm(w)  # Rotation angle

    if theta < 1e-8:
        # Pure translation (or small rotation): use first-order approximation
        T = np.eye(4)
        T[:3, 3] = v
        return T

    # Normalize rotation axis
    w_normalized = w / theta

    # Compute SO(3) exponential (Rodriguez formula)
    w_hat = _skew_symmetric(w_normalized)
    R_matrix = (
        np.eye(3)
        + np.sin(theta) * w_hat
        + (1 - np.cos(theta)) * (w_hat @ w_hat)
    )

    # Compute translation part using the SE(3) formula
    # p = (I - R) * (w × v) + w * (w^T * v) * theta
    # Or equivalently: p = V * v where V is defined below
    V = (
        np.eye(3) * theta
        + (1 - np.cos(theta)) * w_hat
        + (theta - np.sin(theta)) * (w_hat @ w_hat)
    ) / theta

    p = V @ v

    # Construct SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = p

    return T


def _se3_exp_scipy(xi: np.ndarray) -> np.ndarray:
    """
    Alternative implementation using scipy.linalg.expm on the matrix representation.

    This is simpler but potentially slower than the analytical formula.

    Args:
        xi: 6D twist vector [vx, vy, vz, wx, wy, wz]

    Returns:
        4x4 homogeneous transformation matrix in SE(3)
    """
    # Convert twist to 4x4 matrix representation (se(3) matrix)
    xi_hat = _se3_hat(xi)

    # Compute matrix exponential
    return expm(xi_hat)


def _se3_hat(xi: np.ndarray) -> np.ndarray:
    """
    Convert 6D twist vector to its 4x4 matrix representation (hat operator).

    Args:
        xi: 6D twist vector [vx, vy, vz, wx, wy, wz]

    Returns:
        4x4 matrix in se(3) algebra:
        [w^  v]
        [0   0]
        where w^ is the skew-symmetric matrix of angular velocity
    """
    xi = _as_vector(xi, size=6, name="xi")
    v = xi[:3]
    w = xi[3:]

    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = _skew_symmetric(w)
    xi_hat[:3, 3] = v

    return xi_hat


def _skew_symmetric(w: np.ndarray) -> np.ndarray:
    """
    Convert 3D vector to its skew-symmetric matrix (hat operator for so(3)).

    Args:
        w: 3D vector [wx, wy, wz]

    Returns:
        3x3 skew-symmetric matrix:
        [ 0   -wz   wy]
        [ wz   0   -wx]
        [-wy   wx   0 ]
    """
    w = _as_vector(w, size=3, name="w")
    wx, wy, wz = w
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )

@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters for pinhole camera model.
    All units are in pixels unless otherwise specified.

    The primary way to construct this class is via a 3x3 intrinsic matrix K:

        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    Use `CameraIntrinsics.from_K(width, height, K)` or the convenience constructor
    `CameraIntrinsics(width, height, K)` which accepts K directly.
    """

    width: int  # Image width (pixels)
    height: int  # Image height (pixels)
    K: np.ndarray = field(default=None)  # 3x3 intrinsic matrix (primary representation)

    def __post_init__(self):
        if self.K is None:
            # Default: identity-like K with principal point at image center and f=1
            self.K = np.array(
                [
                    [1.0, 0.0, self.width / 2.0],
                    [0.0, 1.0, self.height / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        else:
            self.K = self._validated_K(self.K)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validated_K(K: np.ndarray) -> np.ndarray:
        K = np.asarray(K, dtype=float)
        if K.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix K must be 3x3, got {K.shape}")
        if not np.isclose(K[2, 2], 1.0):
            raise ValueError(f"K[2,2] must be 1, got {K[2, 2]}")
        return K.copy()

    # ------------------------------------------------------------------
    # Derived scalar properties (read from K on the fly)
    # ------------------------------------------------------------------
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_K(cls, width: int, height: int, K: np.ndarray) -> "CameraIntrinsics":
        """Create CameraIntrinsics from image size and a 3x3 intrinsic matrix K."""
        return cls(width=width, height=height, K=K)

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        fov_deg: float = 60.0,
        axis: str = "vertical",
    ) -> "CameraIntrinsics":
        """
        Create CameraIntrinsics from a field-of-view angle.

        Args:
            width:   Image width in pixels.
            height:  Image height in pixels.
            fov_deg: Field-of-view angle in degrees.
            axis:    Whether `fov_deg` describes the 'vertical' or 'horizontal' FOV.
        """
        fov_rad = np.deg2rad(fov_deg)
        if axis == "vertical":
            fy = height / (2.0 * np.tan(fov_rad / 2.0))
            fx = fy
        elif axis == "horizontal":
            fx = width / (2.0 * np.tan(fov_rad / 2.0))
            fy = fx
        else:
            raise ValueError(f"axis must be 'vertical' or 'horizontal', got '{axis}'")

        cx = width / 2.0
        cy = height / 2.0

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
        )
        return cls(width=width, height=height, K=K)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_matrix(self) -> np.ndarray:
        """Return a copy of the 3x3 intrinsic matrix K."""
        return self.K.copy()

    def to_pinhole_parameters(self) -> Any:
        """Convert to Open3D PinholeCameraIntrinsic object."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available.")
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )
        return intrinsic

    def set_K(self, K: np.ndarray) -> None:
        """Replace the intrinsic matrix with a new validated K."""
        self.K = self._validated_K(K)


class VirtualCamera:
    """
    A minimalistic virtual camera class for Open3D scene rendering.

    The canonical way to create a camera is to supply a 3x3 intrinsic matrix K:

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]])
        cam = VirtualCamera.from_K(K, width=1920, height=1080)

    or use the top-level factory:

        cam = create_camera_from_K(K, width=1920, height=1080)

    Features:
    - 6DOF pose representation (x, y, z, pitch, roll, yaw)
    - Pinhole camera model with intrinsic matrix as the primary representation
    - Scene rendering from camera viewpoint
    - Clean API for visual servoing integration
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initialize virtual camera with a CameraIntrinsics object.

        Prefer constructing via `VirtualCamera.from_K(...)` or the module-level
        `create_camera_from_K(...)` factory rather than building CameraIntrinsics
        manually.

        Args:
            intrinsics: CameraIntrinsics built from a K matrix.
        """
        self.intrinsics = intrinsics

        # Initialize pose: position (x, y, z) and orientation (scipy Rotation)
        self._position = np.array([0.0, 0.0, 0.0])  # metres
        self._rotation = R.from_euler("xyz", [0.0, 0.0, 0.0])

        # Cache transformation matrix
        self._update_extrinsics()

        # Persistent renderer to avoid window creation overhead
        self._renderer = None

    # ------------------------------------------------------------------
    # Alternate constructors (K-first API)
    # ------------------------------------------------------------------
    @classmethod
    def from_K(cls, K: np.ndarray, width: int, height: int) -> "VirtualCamera":
        """
        Create a VirtualCamera directly from a 3x3 intrinsic matrix K.

        Args:
            K:      3x3 intrinsic matrix.
            width:  Image width in pixels.
            height: Image height in pixels.

        Returns:
            VirtualCamera instance.
        """
        intrinsics = CameraIntrinsics.from_K(width=width, height=height, K=K)
        return cls(intrinsics)

    @classmethod
    def from_fov(
        cls,
        width: int = 1920,
        height: int = 1080,
        fov_deg: float = 60.0,
        axis: str = "vertical",
    ) -> "VirtualCamera":
        """
        Create a VirtualCamera from a field-of-view angle.

        Internally this builds a K matrix and delegates to `from_K`.

        Args:
            width:   Image width in pixels.
            height:  Image height in pixels.
            fov_deg: Field-of-view in degrees.
            axis:    'vertical' or 'horizontal'.

        Returns:
            VirtualCamera instance.
        """
        intrinsics = CameraIntrinsics.from_fov(
            width=width, height=height, fov_deg=fov_deg, axis=axis
        )
        return cls(intrinsics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_renderer(self) -> Any:
        """Lazy initialization of the offscreen renderer."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available.")
        if self._renderer is None:
            self._renderer = o3d.visualization.rendering.OffscreenRenderer(
                self.intrinsics.width, self.intrinsics.height
            )
        return self._renderer

    def _update_extrinsics(self):
        """Update the extrinsic matrix based on current pose."""
        T = np.eye(4)
        T[:3, :3] = self._rotation.as_matrix()
        T[:3, 3] = self._position
        self._extrinsic_matrix = T

    @property
    def _orientation(self) -> np.ndarray:
        """Get Euler angles from internal rotation."""
        return self._rotation.as_euler("xyz")

    @_orientation.setter
    def _orientation(self, value: np.ndarray):
        """Set internal rotation from Euler angles."""
        self._rotation = R.from_euler("xyz", value)

    # ------------------------------------------------------------------
    # Pose helpers (static)
    # ------------------------------------------------------------------
    @staticmethod
    def _pose_to_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Convert 6DOF pose to 4x4 transformation matrix.

        Args:
            position:    [x, y, z] position vector
            orientation: [pitch, roll, yaw] Euler angles in radians

        Returns:
            4x4 homogeneous transformation matrix (camera-to-world)
        """
        rot = R.from_euler("xyz", orientation)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = position
        return T

    @staticmethod
    def _matrix_to_pose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 4x4 transformation matrix to 6DOF pose.

        Args:
            matrix: 4x4 homogeneous transformation matrix

        Returns:
            Tuple of (position, orientation) where orientation is [pitch, roll, yaw]
        """
        position = matrix[:3, 3]
        rot = R.from_matrix(matrix[:3, :3])
        orientation = rot.as_euler("xyz")
        return position, orientation

    # ------------------------------------------------------------------
    # Public pose API
    # ------------------------------------------------------------------
    def set_pose(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        """
        Set camera pose.

        Args:
            position:    [x, y, z] position in metres (None to keep current)
            orientation: [pitch, roll, yaw] in radians (None to keep current)
        """
        if position is not None:
            self._position = np.array(position, dtype=float)
        if orientation is not None:
            self._orientation = np.array(orientation, dtype=float)
        self._update_extrinsics()

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current camera pose.

        Returns:
            Tuple of (position [x,y,z], orientation [pitch,roll,yaw]).
        """
        return self._position.copy(), self._orientation.copy()

    def get_K(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix K."""
        return self.intrinsics.to_matrix()

    def get_extrinsic_matrix(self) -> np.ndarray:
        """
        Get camera extrinsic matrix (camera-to-world transform).

        Returns:
            4x4 transformation matrix
        """
        return self._extrinsic_matrix.copy()

    def get_view_matrix(self) -> np.ndarray:
        """
        Get view matrix (world-to-camera transform) for rendering.

        Returns:
            4x4 view matrix (inverse of extrinsic)
        """
        return np.linalg.inv(self._extrinsic_matrix)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_rgb(
        self,
        geometry: Any,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Render RGB image from current camera pose.

        Args:
            geometry:         Open3D geometry object(s) to render.
            background_color: RGB background color (0-1 range).

        Returns:
            RGB image as numpy array (H x W x 3) with values in [0, 1].
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available.")
        renderer = self._get_renderer()
        scene = renderer.scene
        scene.clear_geometry()
        scene.set_background(np.array([*background_color, 1.0]))

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        geometries = geometry if isinstance(geometry, list) else [geometry]
        for i, geom in enumerate(geometries):
            scene.add_geometry(f"geometry_{i}", geom, material)

        renderer.setup_camera(
            self.get_K(),
            self.get_view_matrix(),
            self.intrinsics.width,
            self.intrinsics.height,
        )

        image = renderer.render_to_image()
        return np.asarray(image).astype(np.float32) / 255.0

    def render_depth(
        self,
        geometry: Any,
        z_near: float = 0.1,
        z_far: float = 10.0,
    ) -> np.ndarray:
        """
        Render depth image from current camera pose.

        Args:
            geometry: Open3D geometry object(s) to render.
            z_near:   Near clipping plane distance.
            z_far:    Far clipping plane distance.

        Returns:
            Depth image as numpy array (H x W) with values in metres.
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available.")
        renderer = self._get_renderer()
        scene = renderer.scene
        scene.clear_geometry()

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        geometries = geometry if isinstance(geometry, list) else [geometry]
        for i, geom in enumerate(geometries):
            scene.add_geometry(f"geometry_{i}", geom, material)

        renderer.setup_camera(
            self.get_K(),
            self.get_view_matrix(),
            self.intrinsics.width,
            self.intrinsics.height,
        )
        scene.camera.set_projection(
            self.get_K(),
            z_near,
            z_far,
            self.intrinsics.width,
            self.intrinsics.height,
        )

        depth = renderer.render_to_depth_image(z_in_view_space=True)
        return np.asarray(depth)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates using pinhole model.

        Args:
            points_3d: Nx3 array of 3D points in world coordinates.

        Returns:
            Nx2 array of 2D pixel coordinates.
        """
        points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        points_camera = (self.get_view_matrix() @ points_homogeneous.T).T
        points_2d_h = (self.get_K() @ points_camera[:, :3].T).T
        return points_2d_h[:, :2] / points_2d_h[:, 2:3]

    def back_project_points(
        self, points_2d: np.ndarray, depths: np.ndarray
    ) -> np.ndarray:
        """
        Back-project 2D image coordinates to 3D points in camera frame.

        Args:
            points_2d: Nx2 array of pixel coordinates [u, v].
            depths:    N array of depth values (Z in camera frame) in metres.

        Returns:
            Nx3 array of 3D points in camera frame coordinates [X, Y, Z].
        """
        points_2d = np.atleast_2d(points_2d)
        depths = np.atleast_1d(depths)

        if points_2d.shape[0] != depths.shape[0]:
            raise ValueError(
                f"Number of 2D points ({points_2d.shape[0]}) must match "
                f"number of depth values ({depths.shape[0]})"
            )

        K_inv = np.linalg.inv(self.get_K())
        points_2d_h = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])
        normalized = (K_inv @ points_2d_h.T).T
        return normalized * depths[:, np.newaxis]

    def back_project_to_world(
        self, points_2d: np.ndarray, depths: np.ndarray
    ) -> np.ndarray:
        """
        Back-project 2D image coordinates to 3D points in world frame.

        Args:
            points_2d: Nx2 array of pixel coordinates [u, v].
            depths:    N array of depth values in metres.

        Returns:
            Nx3 array of 3D points in world frame coordinates.
        """
        points_camera = self.back_project_points(points_2d, depths)
        points_h = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])
        return (self.get_extrinsic_matrix() @ points_h.T).T[:, :3]

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------
    def look_at(self, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])):
        """
        Orient camera to look at a target point.

        Args:
            target: [x, y, z] target point in world coordinates.
            up:     [x, y, z] up vector (default: z-axis).
        """
        forward = np.asarray(target, dtype=float) - self._position
        norm = np.linalg.norm(forward)
        if norm < 1e-12:
            raise ValueError("Target must be different from camera position")
        forward /= norm

        up = np.asarray(up, dtype=float)
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-12:
            raise ValueError("Up vector must be non-zero")
        up /= up_norm

        if np.abs(np.dot(forward, up)) > 1.0 - 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            if np.abs(np.dot(forward, up)) > 1.0 - 1e-6:
                up = np.array([1.0, 0.0, 0.0])

        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up_corrected = np.cross(forward, right)
        up_corrected /= np.linalg.norm(up_corrected)

        R_world_to_cam = np.vstack([right, up_corrected, forward])
        self._rotation = R.from_matrix(R_world_to_cam.T)
        self._update_extrinsics()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def apply_velocity(
        self, velocity: np.ndarray, dt: float = 1.0, frame: str = "world"
    ) -> None:
        """
            Update camera pose by applying a 6D velocity (velocity) using proper SE(3) exponential map.

            This method integrates the velocity as a single SE(3) element rather than
            separately updating translation and rotation, which properly accounts for
            the coupling between linear and angular motion (screw motion).

            Args:
                velocity: 6D velocity [vx, vy, vz, wx, wy, wz] where:
                       - [vx, vy, vz]: linear velocity (m/s)
                       - [wx, wy, wz]: angular velocity (rad/s)
                dt:    Time step in seconds
                frame: Reference frame: 'world' or 'camera'
                       - 'world': velocity is expressed in world frame
                       - 'camera': velocity is expressed in camera body frame
    """
        velocity = _as_vector(velocity, size=6, name="velocity")
        if frame not in {"world", "camera"}:
            raise ValueError(f"frame must be 'world' or 'camera', got '{frame}'")

        # Scale velocity by time step
        xi = velocity * float(dt)  # ξ ∈ se(3)

        # Compute SE(3) exponential map: exp(ξ^) where ξ^ is the 4x4 matrix representation
        T_delta = _se3_exp(xi)

        # Current pose as SE(3) matrix
        T_current = self.get_extrinsic_matrix()

        # Apply update based on frame
        if frame == "camera":
            # Body-frame velocity: T_new = T_current @ exp(ξ^)
            T_new = T_current @ T_delta
        else:
            # World-frame velocity: T_new = exp(ξ^) @ T_current
            T_new = T_delta @ T_current

        # Extract position and orientation from updated transformation
        self._position = T_new[:3, 3].copy()
        self._rotation = R.from_matrix(T_new[:3, :3])

        # Update cached extrinsic matrix
        self._update_extrinsics()

    # ------------------------------------------------------------------
    # Depth estimation (MoGe)
    # ------------------------------------------------------------------
    def render_depth_MoGe(
        self,
        image: Union[np.ndarray, str, Path],
        model: Optional[Any] = None,
        model_version: str = "v2",
        pretrained_model_name_or_path: Optional[str] = None,
        device_name: str = "cpu",
        use_fp16: bool = False,
        resolution_level: int = 9,
        num_tokens: Optional[int] = None,
        fov_x: Optional[float] = None,
    ) -> np.ndarray:
        """
        Estimate a depth map from a single RGB image using MoGe.

        Args:
            image:                          HxWx3 numpy array or file path.
            model:                          Optional preloaded MoGe model.
            model_version:                  'v1' or 'v2'.
            pretrained_model_name_or_path:  HuggingFace name or local path.
            device_name:                    Torch device string.
            use_fp16:                       Use fp16 inference.
            resolution_level:               MoGe resolution level [0-9].
            num_tokens:                     Optional explicit token count.
            fov_x:                          Optional horizontal FOV in degrees.

        Returns:
            Depth map as HxW float32 numpy array in metres.
        """
        import torch

        if isinstance(image, (str, Path)):
            try:
                import cv2
            except ImportError as exc:
                raise ImportError(
                    "opencv-python is required when image is a file path"
                ) from exc
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.asarray(image)

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"Input image must be (H, W, 3) RGB, got {image_rgb.shape}"
            )

        img = image_rgb.astype(np.float32)
        if np.issubdtype(image_rgb.dtype, np.integer) or img.max() > 1.0:
            img /= 255.0

        device = torch.device(device_name)

        if model is None:
            try:
                from moge.model import import_model_class_by_version
            except ImportError:
                local_moge = Path(__file__).resolve().parents[1] / "MoGe"
                if local_moge.exists() and str(local_moge) not in sys.path:
                    sys.path.insert(0, str(local_moge))
                try:
                    from moge.model import import_model_class_by_version
                except ImportError as exc:
                    raise ImportError(
                        "MoGe is not available. Install it or provide a pre-loaded model."
                    ) from exc

            defaults = {
                "v1": "Ruicheng/moge-vitl",
                "v2": "Ruicheng/moge-2-vitl-normal",
            }
            if pretrained_model_name_or_path is None:
                if model_version not in defaults:
                    raise ValueError(
                        f"Unsupported model version '{model_version}'. "
                        "Expected 'v1' or 'v2'."
                    )
                pretrained_model_name_or_path = defaults[model_version]

            model = (
                import_model_class_by_version(model_version)
                .from_pretrained(pretrained_model_name_or_path)
                .to(device)
                .eval()
            )
            if use_fp16 and hasattr(model, "half"):
                model.half()

        image_tensor = torch.tensor(img, dtype=torch.float32, device=device).permute(
            2, 0, 1
        )
        output = model.infer(
            image_tensor,
            fov_x=fov_x,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16,
        )

        if "depth" not in output:
            raise KeyError('MoGe output is missing key "depth"')

        depth = output["depth"]
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        else:
            depth = np.asarray(depth)

        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2:
            raise ValueError(f"Depth must be (H, W) or (1, H, W); got {depth.shape}")
        if depth.shape != image_rgb.shape[:2]:
            raise ValueError(
                f"Depth shape {depth.shape} != image shape {image_rgb.shape[:2]}"
            )

        return depth.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        pos = self._position
        deg = np.rad2deg(self._orientation)
        K = self.get_K()
        return (
            f"VirtualCamera(\n"
            f"  Position:    [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m\n"
            f"  Orientation: [pitch={deg[0]:.1f}°, roll={deg[1]:.1f}°, yaw={deg[2]:.1f}°]\n"
            f"  Resolution:  {self.intrinsics.width}x{self.intrinsics.height}\n"
            f"  K: [[{K[0, 0]:.1f}, 0, {K[0, 2]:.1f}],\n"
            f"      [0, {K[1, 1]:.1f}, {K[1, 2]:.1f}],\n"
            f"      [0, 0, 1]]\n"
            f")"
        )

    def __str__(self) -> str:
        pos = self._position
        deg = np.rad2deg(self._orientation)
        return (
            f"VirtualCamera at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m, "
            f"orientation [pitch={deg[0]:.1f}°, roll={deg[1]:.1f}°, yaw={deg[2]:.1f}°]"
        )


# ---------------------------------------------------------------------------
# Module-level factory functions (K-first API)
# ---------------------------------------------------------------------------


def create_camera_from_K(
    K: np.ndarray,
    width: int,
    height: int,
) -> VirtualCamera:
    """
    Create a VirtualCamera from a 3x3 intrinsic matrix K.  **Preferred factory.**

    Args:
        K:      3x3 intrinsic matrix.
        width:  Image width in pixels.
        height: Image height in pixels.

    Returns:
        VirtualCamera instance.
    """
    return VirtualCamera.from_K(K=K, width=width, height=height)


def create_default_camera(
    width: int = 1920,
    height: int = 1080,
    fov_deg: float = 60.0,
    intrinsic_matrix: Optional[np.ndarray] = None,
) -> VirtualCamera:
    """
    Create a VirtualCamera with sensible defaults.

    When `intrinsic_matrix` is supplied it is used directly as K.
    Otherwise K is derived from `fov_deg` (vertical FOV).

    Args:
        width:             Image width in pixels.
        height:            Image height in pixels.
        fov_deg:           Vertical field-of-view in degrees (used only when
                           `intrinsic_matrix` is None).
        intrinsic_matrix:  Optional 3x3 K matrix; takes precedence over `fov_deg`.

    Returns:
        VirtualCamera instance.
    """
    if intrinsic_matrix is not None:
        return VirtualCamera.from_K(K=intrinsic_matrix, width=width, height=height)
    return VirtualCamera.from_fov(width=width, height=height, fov_deg=fov_deg)
