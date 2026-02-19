"""
Mesh Scene for Rendering
A class for rendering triangle meshes from arbitrary camera poses.

Dependencies:
    - pyvista: pip install pyvista
    - trimesh: pip install trimesh
    - numpy: pip install numpy
    - torch: pip install torch (optional, for consistency)

Author: Claude
Date: February 2026
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union, List
from pathlib import Path
import pyvista as pv
import trimesh

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MeshScene:
    """
    A triangle mesh scene renderer using PyVista/VTK.

    Features:
    - Load triangle meshes from PLY files
    - Render from arbitrary camera poses
    - Support for vertex colors or textures (via PyVista)
    - Offscreen rendering
    - Integration with VirtualCamera class
    """

    def __init__(
        self,
        verbose: bool = True,
    ):
        """
        Initialize Mesh scene.

        Args:
            verbose: If True, print scene loading details.
        """
        self.verbose = bool(verbose)
        self.mesh: Optional[pv.PolyData] = None
        self._loaded = False
        self.plotter: Optional[pv.Plotter] = None

    def cleanup(self):
        """
        Clear mesh and close plotter.
        """
        self.mesh = None
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None
        self._loaded = False

        if self.verbose:
            print("Mesh scene resources cleaned up.")

    def load_mesh(self, ply_path: str):
        """
        Load a triangle mesh from a PLY file.

        Args:
            ply_path: Path to PLY file
        """
        path = Path(ply_path)
        if not path.exists():
            raise FileNotFoundError(f"PLY file not found: {path}")

        # Use trimesh to load as it handles various PLY formats well
        t_mesh = trimesh.load(str(path))

        if isinstance(t_mesh, trimesh.Scene):
            # If it's a scene, merge into a single mesh
            t_mesh = t_mesh.dump(concatenate=True)

        if not hasattr(t_mesh, "faces") or len(t_mesh.faces) == 0:
            # Check if it might be a point cloud
            if self.verbose:
                print(f"Warning: {path} has no faces. Loading as point cloud.")
            self.mesh = pv.PolyData(t_mesh.vertices)
            if hasattr(t_mesh.visual, "vertex_colors"):
                self.mesh.point_data["colors"] = t_mesh.visual.vertex_colors[:, :3]
        else:
            # Convert to PyVista
            faces = np.column_stack((np.full(len(t_mesh.faces), 3), t_mesh.faces))
            self.mesh = pv.PolyData(t_mesh.vertices, faces)

            # Handle colors
            if hasattr(t_mesh.visual, "vertex_colors"):
                # PyVista expects colors in [0, 255] for uint8 or [0, 1] for float
                colors = t_mesh.visual.vertex_colors[:, :3]
                self.mesh.point_data["colors"] = colors

        self._loaded = True

        if self.verbose:
            print(f"Loaded mesh from {path}")
            print(f"  Vertices: {len(self.mesh.points)}")
            if hasattr(self.mesh, "n_faces"):
                print(f"  Faces: {self.mesh.n_faces}")

    @staticmethod
    def _extract_pose(
        camera_position: np.ndarray, camera_rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (R_c2w, position) from either 3x3 R or 4x4 c2w matrix."""
        if camera_rotation.shape == (4, 4):
            R_c2w = np.asarray(camera_rotation[:3, :3], dtype=np.float64)
            pos = np.asarray(camera_rotation[:3, 3], dtype=np.float64)
        elif camera_rotation.shape == (3, 3):
            R_c2w = np.asarray(camera_rotation, dtype=np.float64)
            pos = np.asarray(camera_position, dtype=np.float64)
        else:
            raise ValueError(f"Invalid rotation shape: {camera_rotation.shape}")
        return R_c2w, pos

    def _configure_plotter_camera_pose(
        self, camera_position: np.ndarray, camera_rotation: np.ndarray
    ) -> None:
        """
        Configure VTK camera axes from c2w pose.

        Convention used by src/camera.py:
        - +X: right
        - +Y: down
        - +Z: forward
        """
        R_c2w, pos = self._extract_pose(camera_position, camera_rotation)
        forward = R_c2w[:, 2]
        # VTK expects a view-up vector for +Y in camera view; negate to map our +Y-down.
        up = -R_c2w[:, 1]
        look_at = pos + forward

        self.plotter.camera.position = pos
        self.plotter.camera.focal_point = look_at
        self.plotter.camera.up = up

    def _configure_plotter_intrinsics(
        self,
        image_width: int,
        image_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        near: float,
        far: float,
    ) -> None:
        """
        Set an explicit VTK projection matrix from pinhole intrinsics.

        This avoids FOV-only approximation and preserves principal point offsets.
        """
        if near <= 0.0:
            raise ValueError(f"near must be > 0, got {near}")
        if far <= near:
            raise ValueError(f"far must be > near, got far={far}, near={near}")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"fx and fy must be > 0, got fx={fx}, fy={fy}")

        n = float(near)
        f = float(far)
        w = float(image_width)
        h = float(image_height)
        fx = float(fx)
        fy = float(fy)
        cx = float(cx)
        cy = float(cy)

        # Frustum bounds for camera convention x-right, y-down, z-forward.
        # VTK camera view uses y-up and -z forward, handled by pose mapping above.
        left = -(w - cx) * n / fx
        right = cx * n / fx
        bottom = -cy * n / fy
        top = (h - cy) * n / fy

        proj = np.array(
            [
                [2.0 * n / (right - left), 0.0, (right + left) / (right - left), 0.0],
                [0.0, 2.0 * n / (top - bottom), (top + bottom) / (top - bottom), 0.0],
                [0.0, 0.0, -(f + n) / (f - n), -(2.0 * f * n) / (f - n)],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

        vtk_proj = pv._vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_proj.SetElement(r, c, float(proj[r, c]))

        self.plotter.camera.SetExplicitProjectionTransformMatrix(vtk_proj)
        self.plotter.camera.UseExplicitProjectionTransformMatrixOn()
        self.plotter.camera.clipping_range = (near, far)

    def render(
        self,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,  # 3x3 rotation matrix or 4x4 extrinsic
        image_width: int,
        image_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        near: float = 0.01,
        far: float = 100.0,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Render the mesh from a given camera pose.

        Uses an explicit VTK projection matrix derived from intrinsics (fx, fy, cx, cy).
        """
        if not self._loaded:
            raise RuntimeError("No mesh loaded. Call load_mesh() first.")

        # Initialize or reuse plotter
        if self.plotter is None or self.plotter.window_size != [
            image_width,
            image_height,
        ]:
            if self.plotter is not None:
                self.plotter.close()
            self.plotter = pv.Plotter(
                off_screen=True, window_size=[image_width, image_height]
            )

        self.plotter.clear()
        self.plotter.background_color = background_color

        if "colors" in self.mesh.point_data:
            self.plotter.add_mesh(
                self.mesh, scalars="colors", rgb=True, preference="point"
            )
        else:
            self.plotter.add_mesh(self.mesh, color="gray")

        # Set up camera
        self._configure_plotter_camera_pose(camera_position, camera_rotation)
        self._configure_plotter_intrinsics(
            image_width=image_width,
            image_height=image_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            near=near,
            far=far,
        )

        # Render
        img = self.plotter.screenshot()

        # Convert to [0, 1] float
        return img.astype(np.float32) / 255.0

    def render_from_virtual_camera(
        self,
        virtual_camera,
        near: float = 0.01,
        far: float = 100.0,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """
        Render from a VirtualCamera object.
        """
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
            background_color=background_color,
        )

    def render_depth(
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
    ) -> np.ndarray:
        """
        Render a depth map from a given camera pose.
        """
        if not self._loaded:
            raise RuntimeError("No mesh loaded. Call load_mesh() first.")

        if self.plotter is None or self.plotter.window_size != [
            image_width,
            image_height,
        ]:
            if self.plotter is not None:
                self.plotter.close()
            self.plotter = pv.Plotter(
                off_screen=True, window_size=[image_width, image_height]
            )

        self.plotter.clear()
        self.plotter.add_mesh(self.mesh)

        self._configure_plotter_camera_pose(camera_position, camera_rotation)
        self._configure_plotter_intrinsics(
            image_width=image_width,
            image_height=image_height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            near=near,
            far=far,
        )

        # Get depth map
        depth = self.plotter.get_image_depth()

        # VTK depth is often normalized or needs conversion to linear depth
        # For now, return what VTK provides.
        return depth

    def render_depth_from_virtual_camera(
        self,
        virtual_camera,
        near: float = 0.01,
        far: float = 100.0,
    ) -> np.ndarray:
        """
        Render depth map from a VirtualCamera object.
        """
        extrinsic = virtual_camera.get_extrinsic_matrix()
        position = extrinsic[:3, 3]
        rotation_matrix = extrinsic[:3, :3]
        intrinsics = virtual_camera.intrinsics

        return self.render_depth(
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
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def render_mesh_view(
    ply_path: str | Path,
    camera_pose: np.ndarray,  # [x, y, z, rx, ry, rz]
    width: int = 1920,
    height: int = 1080,
    fov_deg: float = 60.0,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    High-level function to render a mesh view.
    """
    from src.camera import create_default_camera

    scene = MeshScene()
    scene.load_mesh(str(ply_path))

    camera = create_default_camera(width=width, height=height, fov_deg=fov_deg)
    camera.set_pose(position=camera_pose[:3], orientation=camera_pose[3:])

    rgb = scene.render_from_virtual_camera(camera, background_color=background_color)
    return rgb


if __name__ == "__main__":
    # Test with a simple mesh if possible
    print("Mesh Scene Renderer")
