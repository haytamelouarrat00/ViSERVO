"""
Virtual Camera Class for Open3D
A minimalistic implementation for visual servoing and 3D scene simulation.

Author: Claude
Date: February 2026
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from typing import Any, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for pinhole camera model."""

    width: int
    height: int
    fx: float  # Focal length in x
    fy: float  # Focal length in y
    cx: float  # Principal point x
    cy: float  # Principal point y

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix K."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def to_pinhole_parameters(self) -> o3d.camera.PinholeCameraIntrinsic:
        """Convert to Open3D PinholeCameraIntrinsic object."""
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )
        return intrinsic

    def set_intrinsics_from_K(self, K: np.ndarray) -> None:
        """Set fx/fy/cx/cy from a 3x3 intrinsic matrix K."""
        K = np.asarray(K, dtype=float)
        if K.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix K must be 3x3, got {K.shape}")
        if not np.isclose(K[2, 2], 1.0):
            raise ValueError(f"K[2,2] must be 1, got {K[2,2]}")
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

    @classmethod
    def from_K(cls, width: int, height: int, K: np.ndarray) -> "CameraIntrinsics":
        """Create CameraIntrinsics from image size and a 3x3 intrinsic matrix K."""
        intrinsics = cls(
            width=width,
            height=height,
            fx=1.0,
            fy=1.0,
            cx=width / 2.0,
            cy=height / 2.0,
        )
        intrinsics.set_intrinsics_from_K(K)
        return intrinsics


class VirtualCamera:
    """
    A minimalistic virtual camera class for Open3D scene rendering.

    Features:
    - 6DOF pose representation (x, y, z, pitch, roll, yaw)
    - Pinhole camera model with intrinsic parameters
    - Scene rendering from camera viewpoint
    - Clean API for visual servoing integration
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initialize virtual camera with intrinsic parameters.

        Args:
            intrinsics: CameraIntrinsics object containing camera parameters
        """
        self.intrinsics = intrinsics

        # Initialize pose: position (x, y, z) and orientation (pitch, roll, yaw)
        self._position = np.array([0.0, 0.0, 0.0])  # meters
        self._orientation = np.array([0.0, 0.0, 0.0])  # radians (pitch, roll, yaw)

        # Cache transformation matrix
        self._update_extrinsics()

    def _update_extrinsics(self):
        """Update the extrinsic matrix based on current pose."""
        self._extrinsic_matrix = self._pose_to_matrix(self._position, self._orientation)

    @staticmethod
    def _pose_to_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Convert 6DOF pose to 4x4 transformation matrix.

        Args:
            position: [x, y, z] position vector
            orientation: [pitch, roll, yaw] Euler angles in radians

        Returns:
            4x4 homogeneous transformation matrix (camera-to-world)
        """
        pitch, roll, yaw = orientation

        # Rotation matrices for each axis
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )

        R_y = np.array(
            [
                [np.cos(roll), 0, np.sin(roll)],
                [0, 1, 0],
                [-np.sin(roll), 0, np.cos(roll)],
            ]
        )

        R_z = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        # Combined rotation: R = R_z * R_y * R_x (yaw-roll-pitch order)
        R = R_z @ R_y @ R_x

        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
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
        R = matrix[:3, :3]

        # Extract Euler angles (assuming yaw-roll-pitch order)
        # This is the inverse of the composition in _pose_to_matrix
        roll = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))

        if np.abs(np.cos(roll)) > 1e-6:
            pitch = np.arctan2(R[2, 1] / np.cos(roll), R[2, 2] / np.cos(roll))
            yaw = np.arctan2(R[1, 0] / np.cos(roll), R[0, 0] / np.cos(roll))
        else:
            # Gimbal lock case
            pitch = 0
            yaw = np.arctan2(-R[0, 1], R[1, 1])

        orientation = np.array([pitch, roll, yaw])

        return position, orientation

    def set_pose(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        """
        Set camera pose.

        Args:
            position: [x, y, z] position in meters (None to keep current)
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
            Tuple of (position, orientation) where:
            - position: [x, y, z] in meters
            - orientation: [pitch, roll, yaw] in radians
        """
        return self._position.copy(), self._orientation.copy()

    def get_K(self) -> np.ndarray:
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

    def render_rgb(
        self,
        geometry: Union[o3d.geometry.Geometry, list],
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Render RGB image from current camera pose.

        Args:
            geometry: Open3D geometry object(s) to render (mesh, point cloud, etc.)
            background_color: RGB background color (0-1 range)

        Returns:
            RGB image as numpy array (H x W x 3) with values in [0, 1]
        """
        # Create visualizer in offscreen mode
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=self.intrinsics.width, height=self.intrinsics.height, visible=False
        )

        # Add geometry to scene
        if isinstance(geometry, list):
            for geom in geometry:
                vis.add_geometry(geom)
        else:
            vis.add_geometry(geometry)

        # Set camera parameters
        ctr = vis.get_view_control()
        pinhole = o3d.camera.PinholeCameraParameters()
        pinhole.intrinsic = self.intrinsics.to_pinhole_parameters()
        pinhole.extrinsic = self.get_view_matrix()

        ctr.convert_from_pinhole_camera_parameters(pinhole, allow_arbitrary=True)

        # Set render options
        render_option = vis.get_render_option()
        render_option.background_color = np.array(background_color)

        # Render and capture
        vis.poll_events()
        vis.update_renderer()

        # Capture image
        image = vis.capture_screen_float_buffer(do_render=True)

        vis.destroy_window()

        return np.asarray(image)

    def render_depth(
        self,
        geometry: Union[o3d.geometry.Geometry, list],
        z_near: float = 0.1,
        z_far: float = 10.0,
    ) -> np.ndarray:
        """
        Render depth image from current camera pose.

        Args:
            geometry: Open3D geometry object(s) to render
            z_near: Near clipping plane distance
            z_far: Far clipping plane distance

        Returns:
            Depth image as numpy array (H x W) with values in meters
        """
        # Create visualizer in offscreen mode
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            width=self.intrinsics.width, height=self.intrinsics.height, visible=False
        )

        # Add geometry
        if isinstance(geometry, list):
            for geom in geometry:
                vis.add_geometry(geom)
        else:
            vis.add_geometry(geometry)

        # Set camera parameters
        ctr = vis.get_view_control()
        pinhole = o3d.camera.PinholeCameraParameters()
        pinhole.intrinsic = self.intrinsics.to_pinhole_parameters()
        pinhole.extrinsic = self.get_view_matrix()

        ctr.convert_from_pinhole_camera_parameters(pinhole, allow_arbitrary=True)

        # Render
        vis.poll_events()
        vis.update_renderer()

        # Capture depth
        depth = vis.capture_depth_float_buffer(do_render=True)

        vis.destroy_window()

        return np.asarray(depth)

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates using pinhole model.

        Args:
            points_3d: Nx3 array of 3D points in world coordinates

        Returns:
            Nx2 array of 2D pixel coordinates
        """
        # Transform points to camera coordinates
        points_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        view_matrix = self.get_view_matrix()
        points_camera = (view_matrix @ points_homogeneous.T).T

        # Project to image plane
        K = self.intrinsics.to_matrix()
        points_2d_homogeneous = (K @ points_camera[:, :3].T).T

        # Normalize by depth
        points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]

        return points_2d

    def back_project_points(
        self, points_2d: np.ndarray, depths: np.ndarray
    ) -> np.ndarray:
        """
        Back-project 2D image coordinates to 3D points in camera frame.

        This is the inverse operation of project_points. Given pixel coordinates
        and corresponding depth values, it reconstructs the 3D points in the
        camera coordinate frame.

        Args:
            points_2d: Nx2 array of 2D pixel coordinates [u, v]
            depths: N array of depth values (Z coordinates in camera frame) in meters

        Returns:
            Nx3 array of 3D points in camera frame coordinates [X, Y, Z]

        Example:
            # Given pixel coordinates and depth map
            pixels = np.array([[320, 240], [400, 300]])
            depths = np.array([1.5, 2.0])
            points_3d_cam = camera.back_project_points(pixels, depths)
            # points_3d_cam now contains [X, Y, Z] in camera coordinates
        """
        # Ensure inputs are numpy arrays
        points_2d = np.atleast_2d(points_2d)
        depths = np.atleast_1d(depths)

        if points_2d.shape[0] != depths.shape[0]:
            raise ValueError(
                f"Number of 2D points ({points_2d.shape[0]}) must match "
                f"number of depth values ({depths.shape[0]})"
            )

        # Get intrinsic matrix and compute its inverse
        K = self.intrinsics.to_matrix()
        K_inv = np.linalg.inv(K)

        # Convert pixel coordinates to homogeneous coordinates
        points_2d_homogeneous = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])

        # Back-project to normalized image coordinates
        # [X/Z, Y/Z, 1] = K^-1 @ [u, v, 1]
        normalized_coords = (K_inv @ points_2d_homogeneous.T).T

        # Scale by depth to get 3D coordinates in camera frame
        # [X, Y, Z] = Z * [X/Z, Y/Z, 1]
        points_3d_camera = normalized_coords * depths[:, np.newaxis]

        return points_3d_camera

    def back_project_to_world(
        self, points_2d: np.ndarray, depths: np.ndarray
    ) -> np.ndarray:
        """
        Back-project 2D image coordinates to 3D points in world frame.

        This combines back-projection to camera frame with transformation to world frame.

        Args:
            points_2d: Nx2 array of 2D pixel coordinates [u, v]
            depths: N array of depth values (Z coordinates in camera frame) in meters

        Returns:
            Nx3 array of 3D points in world frame coordinates

        Example:
            # Back-project pixels to world coordinates
            pixels = np.array([[320, 240]])
            depths = np.array([2.0])
            world_points = camera.back_project_to_world(pixels, depths)
        """
        # First back-project to camera frame
        points_camera = self.back_project_points(points_2d, depths)

        # Transform to world frame
        extrinsic = self.get_extrinsic_matrix()
        points_homogeneous = np.hstack(
            [points_camera, np.ones((points_camera.shape[0], 1))]
        )
        points_world = (extrinsic @ points_homogeneous.T).T[:, :3]

        return points_world

    def look_at(self, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])):
        """
        Orient camera to look at a target point.

        Args:
            target: [x, y, z] target point in world coordinates
            up: [x, y, z] up vector (default: z-axis)
        """
        # Compute forward direction (camera +Z points toward the target)
        forward = np.asarray(target, dtype=float) - self._position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-12:
            raise ValueError("Target must be different from camera position")
        forward = forward / forward_norm

        # Normalize up and avoid a degenerate basis when up is collinear with forward
        up = np.asarray(up, dtype=float)
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-12:
            raise ValueError("Up vector must be non-zero")
        up = up / up_norm

        if np.abs(np.dot(forward, up)) > 1.0 - 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            if np.abs(np.dot(forward, up)) > 1.0 - 1e-6:
                up = np.array([1.0, 0.0, 0.0])

        # Build an orthonormal right-handed camera basis with +Z = forward
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(forward, right)
        up_corrected = up_corrected / np.linalg.norm(up_corrected)

        # Build rotation matrix (world to camera)
        R_world_to_cam = np.vstack([right, up_corrected, forward])

        # We need camera to world rotation
        R_cam_to_world = R_world_to_cam.T

        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R_cam_to_world
        T[:3, 3] = self._position

        # Extract orientation from matrix
        _, orientation = self._matrix_to_pose(T)
        self._orientation = orientation

        self._update_extrinsics()

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
            image: Input RGB image as HxWx3 numpy array, or an image file path.
            model: Optional preloaded MoGe model with an `infer(...)` method.
                   If None, the model is loaded from `model_version` and
                   `pretrained_model_name_or_path`.
            model_version: MoGe model version, either "v1" or "v2".
            pretrained_model_name_or_path: Hugging Face model name or local path.
                                           If None, a default model is used.
            device_name: Torch device string (e.g., "cpu", "cuda", "cuda:0").
            use_fp16: Whether to use fp16 inference.
            resolution_level: MoGe inference resolution level [0-9].
            num_tokens: Optional explicit number of inference tokens.
            fov_x: Optional horizontal field of view in degrees.

        Returns:
            Depth map as HxW float32 numpy array in meters.
        """
        import torch

        if isinstance(image, (str, Path)):
            try:
                import cv2
            except ImportError as exc:
                raise ImportError(
                    "opencv-python is required when image is provided as a file path"
                ) from exc

            image_path = Path(image)
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise FileNotFoundError(f"Could not read image file: {image_path}")
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.asarray(image)

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                "Input image must have shape (H, W, 3) in RGB format, "
                f"got {image_rgb.shape}"
            )

        image_normalized = image_rgb.astype(np.float32)
        if np.issubdtype(image_rgb.dtype, np.integer):
            image_normalized = image_normalized / 255.0
        elif image_normalized.max(initial=0.0) > 1.0:
            image_normalized = image_normalized / 255.0

        device = torch.device(device_name)

        if model is None:
            try:
                from moge.model import import_model_class_by_version
            except ImportError:
                # Fallback for local clone layout: project_root/MoGe
                local_moge_repo = Path(__file__).resolve().parents[1] / "MoGe"
                if local_moge_repo.exists() and str(local_moge_repo) not in sys.path:
                    sys.path.insert(0, str(local_moge_repo))
                try:
                    from moge.model import import_model_class_by_version
                except ImportError as exc:
                    raise ImportError(
                        "MoGe is not available. Install it (or use local /MoGe clone), "
                        "or provide a pre-loaded model via `model=`."
                    ) from exc

            default_pretrained = {
                "v1": "Ruicheng/moge-vitl",
                "v2": "Ruicheng/moge-2-vitl-normal",
            }
            if pretrained_model_name_or_path is None:
                if model_version not in default_pretrained:
                    raise ValueError(
                        f"Unsupported model version '{model_version}'. "
                        "Expected one of ['v1', 'v2']."
                    )
                pretrained_model_name_or_path = default_pretrained[model_version]

            model = (
                import_model_class_by_version(model_version)
                .from_pretrained(pretrained_model_name_or_path)
                .to(device)
                .eval()
            )
            if use_fp16 and hasattr(model, "half"):
                model.half()

        image_tensor = torch.tensor(
            image_normalized, dtype=torch.float32, device=device
        ).permute(2, 0, 1)

        output = model.infer(
            image_tensor,
            fov_x=fov_x,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16,
        )

        if "depth" not in output:
            raise KeyError('MoGe inference output is missing key "depth"')

        depth = output["depth"]
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        else:
            depth = np.asarray(depth)

        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]

        if depth.ndim != 2:
            raise ValueError(
                f"Depth map must have shape (H, W) or (1, H, W); got {depth.shape}"
            )

        expected_shape = image_rgb.shape[:2]
        if depth.shape != expected_shape:
            raise ValueError(
                f"Depth map shape {depth.shape} does not match image shape {expected_shape}"
            )

        return depth.astype(np.float32, copy=False)

    def apply_velocity(
        self, velocity: np.ndarray, dt: float = 1.0, frame: str = "world"
    ):
        """
        Update camera pose by applying a 6D velocity vector.

        Args:
            velocity: 6D velocity [vx, vy, vz, wx, wy, wz]
                        First 3 components are linear velocity (m/s)
                        Last 3 components are angular velocity (rad/s)
            dt: Time step for integration (seconds)
            frame: Reference frame for velocity ('world' or 'camera')

        Example:
            # Move forward 0.1m and rotate 0.05 rad around z
            camera.apply_velocity(np.array([0, 0, 0.1, 0, 0, 0.05]), dt=1.0, frame='camera')
        """
        v_lin = velocity[:3]  # Linear velocity
        v_ang = velocity[3:]  # Angular velocity

        # Update position
        if frame == "camera":
            # Transform linear velocity from camera to world frame
            R = self._extrinsic_matrix[:3, :3]
            v_lin_world = R @ v_lin
            self._position += v_lin_world * dt
        else:
            self._position += v_lin * dt

        # Update orientation (simple Euler integration)
        self._orientation += v_ang * dt

        self._update_extrinsics()

    def __repr__(self) -> str:
        """String representation of camera state."""
        pos = self._position
        orient_deg = np.rad2deg(self._orientation)
        return (
            f"VirtualCamera(\n"
            f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m\n"
            f"  Orientation: [pitch={orient_deg[0]:.1f}°, "
            f"roll={orient_deg[1]:.1f}°, yaw={orient_deg[2]:.1f}°]\n"
            f"  Resolution: {self.intrinsics.width}x{self.intrinsics.height}\n"
            f"  Focal: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}\n"
            f")"
        )

    def __str__(self) -> str:
        """Short string representation."""
        pos = self._position
        orient_deg = np.rad2deg(self._orientation)
        return (
            f"VirtualCamera at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m, "
            f"orientation [pitch={orient_deg[0]:.1f}°, roll={orient_deg[1]:.1f}°, "
            f"yaw={orient_deg[2]:.1f}°]"
        )


# Example usage and utility functions
def create_camera_from_K(
    K: np.ndarray,
    width: int,
    height: int,
) -> VirtualCamera:
    """
    Create a VirtualCamera directly from a 3x3 intrinsic matrix K.

    Args:
        K: 3x3 intrinsic matrix.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        VirtualCamera instance using intrinsics from K.
    """
    intrinsics = CameraIntrinsics.from_K(width=width, height=height, K=K)
    return VirtualCamera(intrinsics)


def create_default_camera(
    width: int = 640,
    height: int = 480,
    fov_deg: float = 60.0,
    intrinsic_matrix: Optional[np.ndarray] = None,
    colmap_model_path: str = "data/sparse/0",
    prefer_sfm_intrinsics: bool = True,
) -> VirtualCamera:
    """
    Create a virtual camera with default parameters.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_deg: Vertical field of view in degrees
        intrinsic_matrix: Optional 3x3 intrinsic matrix K. If provided, it overrides
            fx/fy/cx/cy while keeping width/height from function arguments.
        colmap_model_path: Path to COLMAP sparse model folder used to load intrinsics.
        prefer_sfm_intrinsics: If True and intrinsic_matrix is not provided, try loading
            K from COLMAP first; fall back to FOV-based defaults on failure.

    Returns:
        VirtualCamera instance
    """
    # Compute focal length from FOV
    fov_rad = np.deg2rad(fov_deg)
    fy = height / (2 * np.tan(fov_rad / 2))
    fx = fy  # Square pixels

    # Principal point at image center
    cx = width / 2
    cy = height / 2

    intrinsics = CameraIntrinsics(
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy
    )
    K_to_use = intrinsic_matrix
    fallback_reason: Optional[str] = None
    if K_to_use is None and prefer_sfm_intrinsics:
        try:
            from src.utils_colmap import get_camera_intrinsics_and_resolution

            if Path(colmap_model_path).exists():
                sfm_K, sfm_width, sfm_height = get_camera_intrinsics_and_resolution(
                    colmap_model_path
                )
            else:
                sfm_K = None
                fallback_reason = f"COLMAP model path not found: {colmap_model_path}"

            if sfm_K is not None:
                # Scale SfM intrinsics to requested output resolution.
                scale_x = float(width) / float(sfm_width)
                scale_y = float(height) / float(sfm_height)
                K_to_use = np.asarray(sfm_K, dtype=np.float64).copy()
                K_to_use[0, 0] *= scale_x
                K_to_use[1, 1] *= scale_y
                K_to_use[0, 2] *= scale_x
                K_to_use[1, 2] *= scale_y
        except Exception:
            # Fallback to default intrinsics computed from width/height/FOV.
            K_to_use = None
            fallback_reason = (
                f"failed to load intrinsics from COLMAP model: {colmap_model_path}"
            )

    if K_to_use is not None:
        intrinsics.set_intrinsics_from_K(K_to_use)
    elif intrinsic_matrix is None and prefer_sfm_intrinsics:
        print(
            "[create_default_camera] Falling back to default intrinsics "
            f"(fx={intrinsics.fx:.4f}, fy={intrinsics.fy:.4f}, "
            f"cx={intrinsics.cx:.4f}, cy={intrinsics.cy:.4f})"
        )
        if fallback_reason is not None:
            print(f"[create_default_camera] Reason: {fallback_reason}")
        print(f"[create_default_camera] Fallback K:\n{intrinsics.to_matrix()}")

    return VirtualCamera(intrinsics)
