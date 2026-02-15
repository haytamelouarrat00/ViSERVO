"""
Gaussian Splatting Scene for Rendering
A streamlined class for rendering Gaussian Splat scenes from arbitrary camera poses.

Dependencies:
    - gsplat: pip install gsplat
    - torch: pip install torch
    - numpy: pip install numpy

Author: Claude
Date: February 2026
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass

from src.rotations import euler_to_quaternion, quat_to_matrix

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

try:
    from gsplat import rasterization

    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available. Install with: pip install gsplat")


@dataclass
class GaussianParameters:
    """Parameters for a single Gaussian splat or batch of Gaussians."""

    means: np.ndarray  # (N, 3) - 3D positions
    scales: np.ndarray  # (N, 3) - scale in each axis
    rotations: np.ndarray  # (N, 4) - quaternions [w, x, y, z]
    opacities: np.ndarray  # (N, 1) - opacity values [0, 1]
    colors: np.ndarray  # (N, 3) or (N, K, 3) for SH - RGB or spherical harmonics

    def __post_init__(self):
        """Validate shapes."""
        n = len(self.means)
        assert n > 0, "At least one Gaussian is required"
        assert self.means.shape == (n, 3), f"means shape {self.means.shape} != ({n}, 3)"
        assert self.scales.shape == (
            n,
            3,
        ), f"scales shape {self.scales.shape} != ({n}, 3)"
        assert self.rotations.shape == (
            n,
            4,
        ), f"rotations shape {self.rotations.shape} != ({n}, 4)"
        assert self.opacities.shape == (
            n,
            1,
        ), f"opacities shape {self.opacities.shape} != ({n}, 1)"
        # Colors can be (N, 3) for RGB or (N, K, 3) for spherical harmonics
        assert len(self.colors) == n, f"colors length {len(self.colors)} != {n}"

    @property
    def num_gaussians(self) -> int:
        """Return number of Gaussians."""
        return len(self.means)


class GaussianSplattingScene:
    """
    A Gaussian Splatting scene renderer using gsplat library.

    Features:
    - Load Gaussian splat parameters (means, scales, rotations, opacities, colors)
    - Render from arbitrary camera poses
    - Support for RGB and spherical harmonics colors
    - GPU-accelerated rendering via gsplat
    - Integration with VirtualCamera class
    """

    def __init__(
        self,
        device: str = (
            "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        ),
        verbose: bool = True,
    ):
        """
        Initialize Gaussian Splatting scene.

        Args:
            device: Device to use for rendering ('cuda' or 'cpu')
            verbose: If True, print scene loading details.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        if not GSPLAT_AVAILABLE:
            raise ImportError("gsplat is required. Install with: pip install gsplat")

        self.device = torch.device(device)
        self.verbose = bool(verbose)

        # Gaussian parameters (will be set via load_gaussians)
        self.means: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self.rotations: Optional[torch.Tensor] = None
        self.opacities: Optional[torch.Tensor] = None
        self.colors: Optional[torch.Tensor] = None
        self.sh_degree: int = 0  # Spherical harmonics degree (0 for RGB)

        self._loaded = False

    def cleanup(self):
        """
        Clear Gaussian parameters and empty GPU cache.
        """
        self.means = None
        self.scales = None
        self.rotations = None
        self.opacities = None
        self.colors = None
        self._loaded = False

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if self.verbose:
            print("Scene resources cleaned up.")

    def load_gaussians(self, params: GaussianParameters):
        """
        Load Gaussian splat parameters into the scene.

        Args:
            params: GaussianParameters object containing all splat data
        """
        # Ensure previous resources are cleared
        self.cleanup()

        # Convert numpy arrays to torch tensors
        self.means = torch.from_numpy(params.means).float().to(self.device)
        self.scales = torch.from_numpy(params.scales).float().to(self.device)
        self.rotations = torch.from_numpy(params.rotations).float().to(self.device)
        self.opacities = torch.from_numpy(params.opacities).float().to(self.device)
        self.colors = torch.from_numpy(params.colors).float().to(self.device)

        # Detect if using spherical harmonics
        if len(self.colors.shape) == 3:
            # Shape is (N, K, 3) - spherical harmonics
            self.sh_degree = int(np.sqrt(self.colors.shape[1])) - 1
        else:
            # Shape is (N, 3) - RGB colors
            self.sh_degree = 0

        self._loaded = True

        if self.verbose:
            print(f"Loaded {self.num_gaussians} Gaussians")
            print(f"  Device: {self.device}")
            print(f"  SH degree: {self.sh_degree}")

    def load_from_ply(self, ply_path: str):
        """
        Load Gaussian splat parameters from a PLY file.

        Args:
            ply_path: Path to PLY file containing Gaussian splats
        """
        from pathlib import Path
        from plyfile import PlyData

        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        ply = PlyData.read(str(ply_path))
        if "vertex" not in ply:
            raise ValueError(f'PLY file has no "vertex" element: {ply_path}')

        vertex = ply["vertex"].data
        names = set(vertex.dtype.names or [])
        n = len(vertex)
        if n == 0:
            raise ValueError(f"PLY file contains no vertices: {ply_path}")

        def has(*required: str) -> bool:
            return all(k in names for k in required)

        # Positions
        if not has("x", "y", "z"):
            raise ValueError(f"PLY is missing x/y/z fields: {ply_path}")
        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
            np.float32
        )

        # Scales: 3DGS usually stores log-scales in scale_0/1/2
        if has("scale_0", "scale_1", "scale_2"):
            log_scales = np.stack(
                [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
            ).astype(np.float32)
            scales = np.exp(log_scales)
        elif has("sx", "sy", "sz"):
            scales = np.stack(
                [vertex["sx"], vertex["sy"], vertex["sz"]], axis=1
            ).astype(np.float32)
        else:
            scales = np.full((n, 3), 0.01, dtype=np.float32)

        # Rotations as quaternions [w, x, y, z]
        if has("rot_0", "rot_1", "rot_2", "rot_3"):
            rotations = np.stack(
                [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
                axis=1,
            ).astype(np.float32)
        elif has("qw", "qx", "qy", "qz"):
            rotations = np.stack(
                [vertex["qw"], vertex["qx"], vertex["qy"], vertex["qz"]], axis=1
            ).astype(np.float32)
        else:
            rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)

        rot_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
        rot_norm[rot_norm < 1e-12] = 1.0
        rotations = rotations / rot_norm

        # Opacity: 3DGS often stores logits under "opacity"
        if "opacity" in names:
            opacity_raw = np.asarray(vertex["opacity"], dtype=np.float32)
            opacities = 1.0 / (1.0 + np.exp(-opacity_raw))
        elif "alpha" in names:
            opacities = np.asarray(vertex["alpha"], dtype=np.float32)
            if np.max(opacities) > 1.0:
                opacities = opacities / 255.0
        else:
            opacities = np.ones(n, dtype=np.float32)
        opacities = np.clip(opacities, 0.0, 1.0)[:, None]

        # Colors:
        # - 3DGS DC SH coeffs are often stored as f_dc_0/1/2
        #   rgb ~= C0 * f_dc + 0.5, where C0 = 0.28209479177387814
        if has("f_dc_0", "f_dc_1", "f_dc_2"):
            c0 = 0.28209479177387814
            sh_dc = np.stack(
                [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
            ).astype(np.float32)
            colors = np.clip(sh_dc * c0 + 0.5, 0.0, 1.0)
        elif has("red", "green", "blue"):
            colors = np.stack(
                [vertex["red"], vertex["green"], vertex["blue"]], axis=1
            ).astype(np.float32)
            if np.max(colors) > 1.0:
                colors = colors / 255.0
            colors = np.clip(colors, 0.0, 1.0)
        elif has("r", "g", "b"):
            colors = np.stack([vertex["r"], vertex["g"], vertex["b"]], axis=1).astype(
                np.float32
            )
            if np.max(colors) > 1.0:
                colors = colors / 255.0
            colors = np.clip(colors, 0.0, 1.0)
        else:
            colors = np.ones((n, 3), dtype=np.float32) * 0.7

        params = GaussianParameters(
            means=means,
            scales=scales.astype(np.float32),
            rotations=rotations.astype(np.float32),
            opacities=opacities.astype(np.float32),
            colors=colors.astype(np.float32),
        )
        self.load_gaussians(params)
        return params

    @property
    def num_gaussians(self) -> int:
        """Return number of Gaussians in scene."""
        if not self._loaded:
            return 0
        return self.means.shape[0]

    @property
    def is_loaded(self) -> bool:
        """Check if scene has been loaded."""
        return self._loaded

    def render(
        self,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        image_width: int,
        image_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        near: float = 0.01,
        far: float = 100.0,
        return_alpha: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Render the Gaussian splat scene from a given camera pose.

        Args:
            camera_position: Camera position [x, y, z] in world coordinates
            camera_rotation: Camera rotation as quaternion [w, x, y, z] or 3x3 rotation matrix
            image_width: Output image width in pixels
            image_height: Output image height in pixels
            fx: Focal length in x (pixels)
            fy: Focal length in y (pixels)
            cx: Principal point x (pixels)
            cy: Principal point y (pixels)
            near: Near clipping plane
            far: Far clipping plane
            return_alpha: If True, also return alpha channel

        Returns:
            RGB image as numpy array (H, W, 3) with values in [0, 1]
            If return_alpha=True: tuple of (rgb, alpha)
        """
        if not self._loaded:
            raise RuntimeError("No Gaussians loaded. Call load_gaussians() first.")

        # Convert camera parameters to torch tensors
        viewmat = self._build_view_matrix(camera_position, camera_rotation)

        # Build projection matrix
        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )

        # Prepare camera parameters for gsplat
        viewmats = viewmat.unsqueeze(0)  # (1, 4, 4)
        Ks = K.unsqueeze(0)  # (1, 3, 3)
        width = int(image_width)
        height = int(image_height)

        # Rasterize using gsplat
        with torch.no_grad():
            # gsplat.rasterization expects specific format
            renders, alphas, info = rasterization(
                means=self.means,  # (N, 3)
                quats=self.rotations,  # (N, 4) - quaternions
                scales=self.scales,  # (N, 3)
                opacities=self.opacities.squeeze(-1),  # (N,)
                colors=self.colors,  # (N, 3) or (N, K, 3)
                viewmats=viewmats,  # (1, 4, 4)
                Ks=Ks,  # (1, 3, 3)
                width=width,
                height=height,
                near_plane=near,
                far_plane=far,
                sh_degree=self.sh_degree if self.sh_degree > 0 else None,
                render_mode="RGB",
            )

        # Extract rendered image (first in batch)
        rendered_image = renders[0]  # (H, W, 3)

        # Convert to numpy and ensure [0, 1] range
        rgb = rendered_image.cpu().numpy()
        rgb = np.clip(rgb, 0, 1)

        if return_alpha:
            alpha = alphas[0].cpu().numpy()  # (H, W)
            return rgb, alpha

        return rgb

    def render_from_virtual_camera(
        self,
        virtual_camera,
        near: float = 0.01,
        far: float = 100.0,
        return_alpha: bool = False,
    ):
        """
        Render from a VirtualCamera object.

        Args:
            virtual_camera: VirtualCamera instance with set pose
            near: Near clipping plane
            far: Far clipping plane
            return_alpha: If True, also return alpha channel

        Returns:
            RGB image as numpy array (H, W, 3)
            If return_alpha=True: tuple of (rgb, alpha)
        """
        # Use VirtualCamera's exact c2w rotation matrix to avoid convention mismatch
        # between Euler parameterizations in different modules.
        extrinsic = virtual_camera.get_extrinsic_matrix()
        position = extrinsic[:3, 3]
        rotation_matrix = extrinsic[:3, :3]
        intrinsics = virtual_camera.intrinsics

        return self.render(
            camera_position=position,
            camera_rotation=rotation_matrix,
            image_width=intrinsics.width,
            image_height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            near=near,
            far=far,
            return_alpha=return_alpha,
        )

    def render_depth_from_pose(
        self,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        image_width: int,
        image_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        near: float = 0.01,
        far: float = 100.0,
        depth_mode: str = "ED",
        return_alpha: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Render a depth map (z-buffer style) from a given camera pose.

        Args:
            camera_position: Camera position [x, y, z] in world coordinates
            camera_rotation: Camera rotation as quaternion [w, x, y, z] or 3x3 rotation matrix
            image_width: Output image width in pixels
            image_height: Output image height in pixels
            fx: Focal length in x (pixels)
            fy: Focal length in y (pixels)
            cx: Principal point x (pixels)
            cy: Principal point y (pixels)
            near: Near clipping plane
            far: Far clipping plane
            depth_mode: "D" (accumulated depth) or "ED" (expected depth)
            return_alpha: If True, also return alpha channel

        Returns:
            Depth map as numpy array (H, W).
            If return_alpha=True: tuple of (depth, alpha)
        """
        if not self._loaded:
            raise RuntimeError("No Gaussians loaded. Call load_gaussians() first.")
        if depth_mode not in ("D", "ED"):
            raise ValueError(
                f"Unsupported depth_mode '{depth_mode}'. Expected one of ['D', 'ED']."
            )

        viewmat = self._build_view_matrix(camera_position, camera_rotation)
        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )

        viewmats = viewmat.unsqueeze(0)  # (1, 4, 4)
        Ks = K.unsqueeze(0)  # (1, 3, 3)
        width = int(image_width)
        height = int(image_height)

        with torch.no_grad():
            renders, alphas, info = rasterization(
                means=self.means,
                quats=self.rotations,
                scales=self.scales,
                opacities=self.opacities.squeeze(-1),
                colors=self.colors,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                near_plane=near,
                far_plane=far,
                sh_degree=self.sh_degree if self.sh_degree > 0 else None,
                render_mode=depth_mode,
            )

        depth = renders[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = depth.cpu().numpy().astype(np.float32)

        if return_alpha:
            alpha = alphas[0].cpu().numpy().astype(np.float32)
            return depth, alpha
        return depth

    def render_depth_from_virtual_camera(
        self,
        virtual_camera,
        near: float = 0.01,
        far: float = 100.0,
        depth_mode: str = "ED",
        return_alpha: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Render a depth map from a VirtualCamera object.

        Args:
            virtual_camera: VirtualCamera instance with set pose
            near: Near clipping plane
            far: Far clipping plane
            depth_mode: "D" (accumulated depth) or "ED" (expected depth)
            return_alpha: If True, also return alpha channel

        Returns:
            Depth map as numpy array (H, W).
            If return_alpha=True: tuple of (depth, alpha)
        """
        extrinsic = virtual_camera.get_extrinsic_matrix()
        position = extrinsic[:3, 3]
        rotation_matrix = extrinsic[:3, :3]
        intrinsics = virtual_camera.intrinsics

        return self.render_depth_from_pose(
            camera_position=position,
            camera_rotation=rotation_matrix,
            image_width=intrinsics.width,
            image_height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            near=near,
            far=far,
            depth_mode=depth_mode,
            return_alpha=return_alpha,
        )

    def _build_view_matrix(
        self, position: np.ndarray, rotation: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Build view matrix (world to camera transform).

        Args:
            position: Camera position [x, y, z]
            rotation: Either quaternion [w, x, y, z] or 3x3 rotation matrix

        Returns:
            4x4 view matrix as torch tensor
        """
        if isinstance(rotation, np.ndarray):
            if rotation.shape == (4,):
                # Quaternion to rotation matrix
                R = self._quat_to_matrix(rotation)
            elif rotation.shape == (3, 3):
                R = rotation
            else:
                raise ValueError(f"Invalid rotation shape: {rotation.shape}")
        else:
            raise ValueError(f"Invalid rotation type: {type(rotation)}")

        # Build camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = position

        # Invert to get world-to-camera (view matrix)
        w2c = np.linalg.inv(c2w)

        return torch.from_numpy(w2c).float().to(self.device)

    @staticmethod
    def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            3x3 rotation matrix
        """
        return quat_to_matrix(quat)

    @staticmethod
    def _euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to quaternion.

        Args:
            euler: Euler angles [pitch, roll, yaw] in radians

        Returns:
            Quaternion [w, x, y, z]
        """
        return euler_to_quaternion(euler)

    def get_info(self) -> Dict[str, any]:
        """
        Get information about the loaded scene.

        Returns:
            Dictionary with scene information
        """
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "num_gaussians": self.num_gaussians,
            "device": str(self.device),
            "sh_degree": self.sh_degree,
            "means_range": {
                "min": self.means.min(dim=0)[0].cpu().numpy().tolist(),
                "max": self.means.max(dim=0)[0].cpu().numpy().tolist(),
            },
            "scale_range": {
                "min": self.scales.min(dim=0)[0].cpu().numpy().tolist(),
                "max": self.scales.max(dim=0)[0].cpu().numpy().tolist(),
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        if not self._loaded:
            return "GaussianSplattingScene(not loaded)"

        return (
            f"GaussianSplattingScene(\n"
            f"  num_gaussians={self.num_gaussians}\n"
            f"  device={self.device}\n"
            f"  sh_degree={self.sh_degree}\n"
            f")"
        )


# Utility functions for creating test scenes


def create_simple_test_scene(
    num_gaussians: int = 100,
    center: Tuple[float, float, float] = (0, 0, 0),
    radius: float = 1.0,
    seed: Optional[int] = 42,
) -> GaussianParameters:
    """
    Create a simple test scene with randomly positioned Gaussians.

    Args:
        num_gaussians: Number of Gaussians to create
        center: Center position of the scene
        radius: Radius of the scene
        seed: Random seed for reproducibility

    Returns:
        GaussianParameters object
    """
    if seed is not None:
        np.random.seed(seed)

    # Random positions within sphere
    theta = np.random.uniform(0, 2 * np.pi, num_gaussians)
    phi = np.random.uniform(0, np.pi, num_gaussians)
    r = np.random.uniform(0, radius, num_gaussians)

    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] + r * np.cos(phi)

    means = np.stack([x, y, z], axis=1)

    # Small random scales
    scales = np.random.uniform(0.01, 0.05, (num_gaussians, 3))

    # Random rotations (quaternions)
    # Generate random unit quaternions
    rotations = np.random.randn(num_gaussians, 4)
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

    # Random opacities
    opacities = np.random.uniform(0.5, 1.0, (num_gaussians, 1))

    # Random colors (RGB)
    colors = np.random.uniform(0, 1, (num_gaussians, 3))

    return GaussianParameters(
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
    )


def create_colored_cube_scene(
    num_gaussians_per_face: int = 50, cube_size: float = 1.0
) -> GaussianParameters:
    """
    Create a colored cube scene with Gaussians on each face.

    Args:
        num_gaussians_per_face: Number of Gaussians per cube face
        cube_size: Size of the cube

    Returns:
        GaussianParameters object
    """
    if num_gaussians_per_face <= 0:
        raise ValueError("num_gaussians_per_face must be positive")

    half_size = cube_size / 2

    all_means = []
    all_colors = []

    # Define 6 faces with their positions and colors
    faces = [
        # (normal, color, position_offset)
        ([0, 0, 1], [1, 0, 0], [0, 0, half_size]),  # Front - Red
        ([0, 0, -1], [0, 1, 0], [0, 0, -half_size]),  # Back - Green
        ([1, 0, 0], [0, 0, 1], [half_size, 0, 0]),  # Right - Blue
        ([-1, 0, 0], [1, 1, 0], [-half_size, 0, 0]),  # Left - Yellow
        ([0, 1, 0], [1, 0, 1], [0, half_size, 0]),  # Top - Magenta
        ([0, -1, 0], [0, 1, 1], [0, -half_size, 0]),  # Bottom - Cyan
    ]

    for normal, color, offset in faces:
        # Create grid on this face
        n = int(np.ceil(np.sqrt(num_gaussians_per_face)))
        u = np.linspace(-half_size, half_size, n)
        v = np.linspace(-half_size, half_size, n)
        uu, vv = np.meshgrid(u, v)

        # Determine which axes to use based on normal
        if abs(normal[2]) > 0:  # Front/Back face (XY plane)
            face_means = np.stack(
                [uu.flatten(), vv.flatten(), np.full(n * n, offset[2])], axis=1
            )
        elif abs(normal[0]) > 0:  # Left/Right face (YZ plane)
            face_means = np.stack(
                [np.full(n * n, offset[0]), uu.flatten(), vv.flatten()], axis=1
            )
        else:  # Top/Bottom face (XZ plane)
            face_means = np.stack(
                [uu.flatten(), np.full(n * n, offset[1]), vv.flatten()], axis=1
            )

        face_means = face_means[:num_gaussians_per_face]
        all_means.append(face_means)
        all_colors.append(np.tile(color, (num_gaussians_per_face, 1)))

    means = np.vstack(all_means)
    colors = np.vstack(all_colors)

    num_total = len(means)

    # Small uniform scales
    scales = np.full((num_total, 3), 0.02)

    # Identity rotations
    rotations = np.tile([1, 0, 0, 0], (num_total, 1))

    # Full opacity
    opacities = np.ones((num_total, 1))

    return GaussianParameters(
        means=means,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
    )


if __name__ == "__main__":
    print("Gaussian Splatting Scene Renderer")
    print("=" * 50)
    print("\nThis module provides Gaussian Splatting rendering")
    print("using the gsplat library.")
    print("\nKey Features:")
    print("  • Load Gaussian splat parameters")
    print("  • Render from arbitrary camera poses")
    print("  • GPU-accelerated rasterization")
    print("  • Integration with VirtualCamera")
    print("\nRequirements:")
    print("  - PyTorch: pip install torch")
    print("  - gsplat: pip install gsplat")
    print("\nSee gaussian_scene_example.py for usage examples.")
