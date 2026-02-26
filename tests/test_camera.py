"""
Unit tests for the VirtualCamera module.

Covers:
- Helper functions: _as_vector, _skew_symmetric, _se3_hat, _se3_exp
- CameraIntrinsics: construction, validation, derived properties, FOV constructor
- VirtualCamera: pose API, projection/back-projection, look_at, apply_velocity
- Module-level factories: create_camera_from_K, create_default_camera

Run with:
    pytest test_virtual_camera.py -v

Dependencies:
    pip install pytest numpy scipy
    (open3d is mocked wherever needed)
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Stub out open3d so tests run without it installed
# ---------------------------------------------------------------------------
_o3d_stub = types.ModuleType("open3d")
_o3d_stub.camera = MagicMock()
_o3d_stub.visualization = MagicMock()
sys.modules.setdefault("open3d", _o3d_stub)

# Now import the module under test
from src.camera import (  # noqa: E402  (import after sys.modules patch)
    CameraIntrinsics,
    VirtualCamera,
    _as_vector,
    _se3_exp,
    _se3_exp_scipy,
    _se3_hat,
    _skew_symmetric,
    create_camera_from_K,
    create_default_camera,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_K():
    """A realistic 3x3 intrinsic matrix."""
    return np.array([
        [800.0,   0.0, 320.0],
        [  0.0, 800.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])


@pytest.fixture
def default_intrinsics(default_K):
    return CameraIntrinsics(width=640, height=480, K=default_K)


@pytest.fixture
def default_camera(default_intrinsics):
    return VirtualCamera(default_intrinsics)


# ===========================================================================
# _as_vector
# ===========================================================================

class TestAsVector:
    def test_list_converted_correctly(self):
        result = _as_vector([1, 2, 3], size=3, name="v")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        assert result.dtype == float

    def test_2d_array_flattened(self):
        result = _as_vector(np.array([[1, 2, 3]]), size=3, name="v")
        assert result.shape == (3,)

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="must have 3 elements"):
            _as_vector([1, 2], size=3, name="v")

    def test_scalar_raises(self):
        with pytest.raises(ValueError):
            _as_vector(5.0, size=3, name="v")

    def test_six_element_vector(self):
        xi = _as_vector(list(range(6)), size=6, name="xi")
        assert xi.shape == (6,)


# ===========================================================================
# _skew_symmetric
# ===========================================================================

class TestSkewSymmetric:
    def test_known_result(self):
        w = np.array([1.0, 2.0, 3.0])
        S = _skew_symmetric(w)
        expected = np.array([
            [ 0, -3,  2],
            [ 3,  0, -1],
            [-2,  1,  0],
        ], dtype=float)
        np.testing.assert_allclose(S, expected)

    def test_skew_property(self):
        """S + S^T == 0."""
        S = _skew_symmetric(np.array([4.0, -1.0, 0.5]))
        np.testing.assert_allclose(S + S.T, np.zeros((3, 3)), atol=1e-12)

    def test_zero_vector_gives_zero_matrix(self):
        S = _skew_symmetric(np.zeros(3))
        np.testing.assert_array_equal(S, np.zeros((3, 3)))

    def test_cross_product_equivalence(self):
        """S(w) @ v == w × v."""
        w = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        S = _skew_symmetric(w)
        np.testing.assert_allclose(S @ v, np.cross(w, v), atol=1e-12)

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError):
            _skew_symmetric([1, 2])


# ===========================================================================
# _se3_hat
# ===========================================================================

class TestSe3Hat:
    def test_shape(self):
        xi = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        assert _se3_hat(xi).shape == (4, 4)

    def test_structure(self):
        """Top-left 3x3 is skew-symmetric; top-right column is v; bottom row is zeros."""
        xi = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        H = _se3_hat(xi)
        np.testing.assert_allclose(H[:3, :3] + H[:3, :3].T, np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(H[:3, 3], xi[:3])
        np.testing.assert_allclose(H[3, :], np.zeros(4))

    def test_pure_translation(self):
        xi = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        H = _se3_hat(xi)
        np.testing.assert_allclose(H[:3, :3], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(H[:3, 3], [5.0, 0.0, 0.0])


# ===========================================================================
# _se3_exp
# ===========================================================================

class TestSe3Exp:
    def test_zero_twist_gives_identity(self):
        T = _se3_exp(np.zeros(6))
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_pure_translation(self):
        xi = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        T = _se3_exp(xi)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-10)

    def test_pure_rotation_z(self):
        """90° rotation about z-axis, no translation."""
        angle = np.pi / 2
        xi = np.array([0.0, 0.0, 0.0, 0.0, 0.0, angle])
        T = _se3_exp(xi)
        expected_R = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], np.zeros(3), atol=1e-10)

    def test_result_is_proper_se3(self):
        """det(R) == 1 and bottom row is [0, 0, 0, 1]."""
        xi = np.array([0.5, -0.3, 0.1, 0.2, -0.4, 0.6])
        T = _se3_exp(xi)
        assert T.shape == (4, 4)
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_agrees_with_scipy_implementation(self):
        """Analytical and scipy implementations must agree."""
        xi = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.5])
        T_analytical = _se3_exp(xi)
        T_scipy = _se3_exp_scipy(xi)
        np.testing.assert_allclose(T_analytical, T_scipy, atol=1e-8)

    def test_small_rotation_near_identity(self):
        xi = np.array([0.0, 0.0, 0.0, 1e-10, 1e-10, 1e-10])
        T = _se3_exp(xi)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-8)

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError):
            _se3_exp([1, 2, 3])


# ===========================================================================
# CameraIntrinsics
# ===========================================================================

class TestCameraIntrinsics:

    # --- Construction ---

    def test_default_K_when_none(self):
        ci = CameraIntrinsics(width=640, height=480)
        assert ci.K.shape == (3, 3)
        assert np.isclose(ci.K[2, 2], 1.0)
        assert np.isclose(ci.cx, 320.0)
        assert np.isclose(ci.cy, 240.0)

    def test_from_K_constructor(self, default_K):
        ci = CameraIntrinsics.from_K(640, 480, default_K)
        np.testing.assert_allclose(ci.K, default_K)

    def test_K_is_copied(self, default_K):
        ci = CameraIntrinsics(width=640, height=480, K=default_K)
        default_K[0, 0] = 9999.0
        assert ci.fx != 9999.0  # original must not be mutated

    # --- Derived properties ---

    def test_scalar_properties(self, default_K):
        ci = CameraIntrinsics(width=640, height=480, K=default_K)
        assert ci.fx == 800.0
        assert ci.fy == 800.0
        assert ci.cx == 320.0
        assert ci.cy == 240.0

    # --- Validation ---

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="3x3"):
            CameraIntrinsics(width=640, height=480, K=np.eye(2))

    def test_K22_not_one_raises(self):
        bad_K = np.eye(3) * 2.0
        with pytest.raises(ValueError, match="K\[2,2\] must be 1"):
            CameraIntrinsics(width=640, height=480, K=bad_K)

    # --- from_fov ---

    def test_from_fov_vertical(self):
        ci = CameraIntrinsics.from_fov(width=640, height=480, fov_deg=60.0, axis="vertical")
        expected_fy = 480 / (2 * np.tan(np.deg2rad(60) / 2))
        assert np.isclose(ci.fy, expected_fy)

    def test_from_fov_horizontal(self):
        ci = CameraIntrinsics.from_fov(width=640, height=480, fov_deg=90.0, axis="horizontal")
        expected_fx = 640 / (2 * np.tan(np.deg2rad(90) / 2))
        assert np.isclose(ci.fx, expected_fx)

    def test_from_fov_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="axis must be"):
            CameraIntrinsics.from_fov(640, 480, fov_deg=60.0, axis="diagonal")

    # --- to_matrix ---

    def test_to_matrix_returns_copy(self, default_intrinsics):
        m = default_intrinsics.to_matrix()
        m[0, 0] = 9999.0
        assert default_intrinsics.fx != 9999.0

    # --- set_K ---

    def test_set_K_updates_properties(self, default_intrinsics):
        new_K = np.array([
            [500.0, 0.0, 100.0],
            [0.0, 500.0, 200.0],
            [0.0,   0.0,   1.0],
        ])
        default_intrinsics.set_K(new_K)
        assert default_intrinsics.fx == 500.0

    def test_set_K_invalid_raises(self, default_intrinsics):
        with pytest.raises(ValueError):
            default_intrinsics.set_K(np.eye(4))

    # --- to_pinhole_parameters without open3d ---

    def test_to_pinhole_parameters_no_open3d(self, default_intrinsics, monkeypatch):
        import src.camera as vc_module
        monkeypatch.setattr(vc_module, "OPEN3D_AVAILABLE", False)
        with pytest.raises(ImportError):
            default_intrinsics.to_pinhole_parameters()


# ===========================================================================
# VirtualCamera – construction
# ===========================================================================

class TestVirtualCameraConstruction:

    def test_from_K(self, default_K):
        cam = VirtualCamera.from_K(default_K, width=640, height=480)
        np.testing.assert_allclose(cam.get_K(), default_K)
        assert cam.intrinsics.width == 640

    def test_from_fov_creates_camera(self):
        cam = VirtualCamera.from_fov(width=320, height=240, fov_deg=90.0)
        assert cam.intrinsics.width == 320

    def test_initial_pose_is_origin(self, default_camera):
        pos, ori = default_camera.get_pose()
        np.testing.assert_allclose(pos, [0, 0, 0])
        np.testing.assert_allclose(ori, [0, 0, 0], atol=1e-12)

    def test_initial_extrinsic_is_identity(self, default_camera):
        np.testing.assert_allclose(default_camera.get_extrinsic_matrix(), np.eye(4))

    def test_repr_contains_position(self, default_camera):
        assert "Position" in repr(default_camera)

    def test_str_contains_position(self, default_camera):
        assert "VirtualCamera at" in str(default_camera)


# ===========================================================================
# VirtualCamera – pose API
# ===========================================================================

class TestVirtualCameraPose:

    def test_set_position(self, default_camera):
        default_camera.set_pose(position=[1.0, 2.0, 3.0])
        pos, _ = default_camera.get_pose()
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0])

    def test_set_orientation(self, default_camera):
        angles = [0.1, 0.2, 0.3]
        default_camera.set_pose(orientation=angles)
        _, ori = default_camera.get_pose()
        np.testing.assert_allclose(ori, angles, atol=1e-10)

    def test_set_pose_none_keeps_existing(self, default_camera):
        default_camera.set_pose(position=[5.0, 0.0, 0.0])
        default_camera.set_pose(orientation=[0.1, 0.0, 0.0])  # position unchanged
        pos, _ = default_camera.get_pose()
        np.testing.assert_allclose(pos, [5.0, 0.0, 0.0])

    def test_get_pose_returns_copies(self, default_camera):
        pos, ori = default_camera.get_pose()
        pos[:] = 99.0
        pos2, _ = default_camera.get_pose()
        assert not np.all(pos2 == 99.0)

    def test_extrinsic_reflects_pose(self, default_camera):
        default_camera.set_pose(position=[1.0, 2.0, 3.0])
        T = default_camera.get_extrinsic_matrix()
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])

    def test_view_matrix_is_inverse_of_extrinsic(self, default_camera):
        default_camera.set_pose(position=[1.0, -0.5, 2.0], orientation=[0.1, 0.2, 0.3])
        T = default_camera.get_extrinsic_matrix()
        V = default_camera.get_view_matrix()
        np.testing.assert_allclose(T @ V, np.eye(4), atol=1e-10)

    def test_get_extrinsic_returns_copy(self, default_camera):
        T = default_camera.get_extrinsic_matrix()
        T[0, 3] = 9999.0
        T2 = default_camera.get_extrinsic_matrix()
        assert T2[0, 3] != 9999.0


# ===========================================================================
# VirtualCamera – projection
# ===========================================================================

class TestProjection:

    @pytest.fixture
    def cam_at_origin(self, default_K):
        """Camera at origin looking along +z."""
        cam = VirtualCamera.from_K(default_K, width=640, height=480)
        return cam

    def test_project_single_point_on_axis(self, cam_at_origin):
        """Point directly in front of camera should land at principal point."""
        K = cam_at_origin.get_K()
        cx, cy = K[0, 2], K[1, 2]
        # Camera at origin, identity pose → view matrix = identity
        point = np.array([[0.0, 0.0, 1.0]])  # 1 m in front
        px = cam_at_origin.project_points(point)
        np.testing.assert_allclose(px, [[cx, cy]], atol=1e-8)

    def test_project_multiple_points(self, cam_at_origin):
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
        px = cam_at_origin.project_points(points)
        assert px.shape == (2, 2)

    def test_back_project_roundtrip(self, cam_at_origin):
        """project → back-project should recover original point."""
        K = cam_at_origin.get_K()
        cx, cy = K[0, 2], K[1, 2]
        depth = 2.0
        # Project a camera-frame point to 2D
        p_cam = np.array([[0.0, 0.0, depth]])
        px = cam_at_origin.project_points(p_cam)
        # Back-project
        recovered = cam_at_origin.back_project_points(px, np.array([depth]))
        np.testing.assert_allclose(recovered, p_cam, atol=1e-8)

    def test_back_project_depth_mismatch_raises(self, cam_at_origin):
        with pytest.raises(ValueError, match="must match"):
            cam_at_origin.back_project_points(
                np.array([[100.0, 200.0], [150.0, 250.0]]),
                np.array([1.0])
            )

    def test_back_project_to_world_at_origin(self, cam_at_origin):
        """With identity pose, camera frame == world frame."""
        K = cam_at_origin.get_K()
        cx, cy = K[0, 2], K[1, 2]
        depth = 3.0
        pts2d = np.array([[cx, cy]])
        world = cam_at_origin.back_project_to_world(pts2d, np.array([depth]))
        # Should be along z-axis
        np.testing.assert_allclose(world, [[0.0, 0.0, depth]], atol=1e-8)

    def test_back_project_single_2d_point(self, cam_at_origin):
        pts = np.array([320.0, 240.0])  # 1D, not 2D
        result = cam_at_origin.back_project_points(pts.reshape(1, 2), np.array([1.0]))
        assert result.shape == (1, 3)


# ===========================================================================
# VirtualCamera – look_at
# ===========================================================================

class TestLookAt:

    def test_look_at_updates_rotation(self, default_camera):
        default_camera.set_pose(position=[0.0, 0.0, 0.0])
        default_camera.look_at([0.0, 0.0, 5.0])
        # Forward direction should now align with +z
        T = default_camera.get_extrinsic_matrix()
        forward = T[:3, 2]
        np.testing.assert_allclose(forward, [0.0, 0.0, 1.0], atol=1e-10)

    def test_look_at_zero_target_raises(self, default_camera):
        default_camera.set_pose(position=[0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="different from camera position"):
            default_camera.look_at([0.0, 0.0, 0.0])

    def test_look_at_zero_up_raises(self, default_camera):
        with pytest.raises(ValueError, match="non-zero"):
            default_camera.look_at([1.0, 0.0, 0.0], up=[0.0, 0.0, 0.0])

    def test_look_at_updates_extrinsic(self, default_camera):
        default_camera.look_at([1.0, 0.0, 0.0])
        T = default_camera.get_extrinsic_matrix()
        assert T.shape == (4, 4)
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10


# ===========================================================================
# VirtualCamera – apply_velocity
# ===========================================================================

class TestApplyVelocity:

    def test_pure_translation_world_frame(self, default_camera):
        velocity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        default_camera.apply_velocity(velocity, dt=1.0, frame="world")
        pos, _ = default_camera.get_pose()
        np.testing.assert_allclose(pos, [1.0, 0.0, 0.0], atol=1e-8)

    def test_zero_velocity_no_change(self, default_camera):
        default_camera.apply_velocity(np.zeros(6), dt=1.0, frame="world")
        pos, ori = default_camera.get_pose()
        np.testing.assert_allclose(pos, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(ori, [0, 0, 0], atol=1e-10)

    def test_camera_frame_translation(self, default_camera):
        """Moving along camera +z in body frame with identity pose == world +z."""
        velocity = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        default_camera.apply_velocity(velocity, dt=1.0, frame="camera")
        pos, _ = default_camera.get_pose()
        np.testing.assert_allclose(pos, [0.0, 0.0, 1.0], atol=1e-8)

    def test_dt_scales_displacement(self, default_camera):
        velocity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        default_camera.apply_velocity(velocity, dt=2.0, frame="world")
        pos, _ = default_camera.get_pose()
        np.testing.assert_allclose(pos[0], 2.0, atol=1e-8)

    def test_invalid_frame_raises(self, default_camera):
        with pytest.raises(ValueError, match="frame must be"):
            default_camera.apply_velocity(np.zeros(6), dt=1.0, frame="robot")

    def test_invalid_velocity_size_raises(self, default_camera):
        with pytest.raises(ValueError):
            default_camera.apply_velocity([1.0, 2.0, 3.0], dt=1.0)

    def test_pure_rotation_changes_orientation(self, default_camera):
        """Applying angular velocity should change orientation, not position."""
        velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
        default_camera.apply_velocity(velocity, dt=1.0, frame="world")
        pos, ori = default_camera.get_pose()
        np.testing.assert_allclose(pos, [0, 0, 0], atol=1e-8)
        # yaw (ori[2]) should be approximately pi/2
        assert abs(ori[2]) > 0.1  # some rotation happened

    def test_extrinsic_stays_valid_se3(self, default_camera):
        velocity = np.array([0.3, -0.1, 0.5, 0.2, 0.1, -0.3])
        default_camera.apply_velocity(velocity, dt=0.1, frame="camera")
        T = default_camera.get_extrinsic_matrix()
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)


# ===========================================================================
# Module-level factories
# ===========================================================================

class TestFactories:

    def test_create_camera_from_K(self, default_K):
        cam = create_camera_from_K(default_K, width=640, height=480)
        assert isinstance(cam, VirtualCamera)
        np.testing.assert_allclose(cam.get_K(), default_K)

    def test_create_default_camera_fov(self):
        cam = create_default_camera(width=1920, height=1080, fov_deg=60.0)
        assert cam.intrinsics.width == 1920
        assert cam.intrinsics.height == 1080

    def test_create_default_camera_with_K(self, default_K):
        cam = create_default_camera(width=640, height=480, intrinsic_matrix=default_K)
        np.testing.assert_allclose(cam.get_K(), default_K)

    def test_create_default_camera_K_takes_precedence(self, default_K):
        """When K is provided, fov_deg is ignored."""
        cam_with_K = create_default_camera(
            width=640, height=480, fov_deg=90.0, intrinsic_matrix=default_K
        )
        cam_fov_only = create_default_camera(width=640, height=480, fov_deg=90.0)
        # focal lengths should differ because K overrides fov
        assert not np.isclose(cam_with_K.intrinsics.fx, cam_fov_only.intrinsics.fx)


# ===========================================================================
# Rendering – Open3D unavailable path
# ===========================================================================

class TestRenderingNoOpen3D:

    def test_render_rgb_raises_without_open3d(self, default_camera, monkeypatch):
        import src.camera as vc_module
        monkeypatch.setattr(vc_module, "OPEN3D_AVAILABLE", False)
        with pytest.raises(ImportError):
            default_camera.render_rgb(MagicMock())

    def test_render_depth_raises_without_open3d(self, default_camera, monkeypatch):
        import src.camera as vc_module
        monkeypatch.setattr(vc_module, "OPEN3D_AVAILABLE", False)
        with pytest.raises(ImportError):
            default_camera.render_depth(MagicMock())


# ===========================================================================
# render_depth_MoGe – input validation (no model required)
# ===========================================================================

class TestRenderDepthMoGe:

    def test_wrong_image_shape_raises(self, default_camera):
        """Grayscale image must raise ValueError."""
        bad_image = np.zeros((480, 640), dtype=np.uint8)  # 2-D, not 3-channel
        with pytest.raises((ValueError, ImportError)):
            default_camera.render_depth_MoGe(bad_image)

    def test_file_not_found_raises(self, default_camera):
        with pytest.raises((FileNotFoundError, ImportError, ModuleNotFoundError)):
            default_camera.render_depth_MoGe("/nonexistent/path/image.png")

    def test_with_mock_model(self, default_camera):
        """Happy path with a fully mocked MoGe model."""
        pytest.importorskip("torch")
        import torch

        H, W = 480, 640
        image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        fake_depth = torch.ones((H, W), dtype=torch.float32)

        mock_model = MagicMock()
        mock_model.infer.return_value = {"depth": fake_depth}

        depth = default_camera.render_depth_MoGe(image, model=mock_model)
        assert depth.shape == (H, W)
        assert depth.dtype == np.float32

    def test_model_missing_depth_key_raises(self, default_camera):
        pytest.importorskip("torch")
        H, W = 480, 640
        image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

        mock_model = MagicMock()
        mock_model.infer.return_value = {}  # missing 'depth'

        with pytest.raises(KeyError, match="depth"):
            default_camera.render_depth_MoGe(image, model=mock_model)


# ===========================================================================
# Pose helpers (static methods)
# ===========================================================================

class TestPoseHelpers:

    def test_pose_to_matrix_identity(self):
        T = VirtualCamera._pose_to_matrix(
            np.zeros(3), np.zeros(3)
        )
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_matrix_to_pose_roundtrip(self):
        pos_in = np.array([1.0, -2.0, 0.5])
        ori_in = np.array([0.1, -0.2, 0.3])
        T = VirtualCamera._pose_to_matrix(pos_in, ori_in)
        pos_out, ori_out = VirtualCamera._matrix_to_pose(T)
        np.testing.assert_allclose(pos_out, pos_in, atol=1e-10)
        np.testing.assert_allclose(ori_out, ori_in, atol=1e-10)

    def test_matrix_to_pose_identity(self):
        pos, ori = VirtualCamera._matrix_to_pose(np.eye(4))
        np.testing.assert_allclose(pos, [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(ori, [0, 0, 0], atol=1e-12)

    