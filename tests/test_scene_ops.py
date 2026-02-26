import numpy as np
import pytest
from src.scene_ops import scene_bounds, quat_multiply_left, apply_scene_pose, pose6_from_T
from src.gs import GaussianParameters
from scipy.spatial.transform import Rotation as R

def test_scene_bounds():
    points = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [-1, -1, -1]
    ])
    mins, maxs, center, radius = scene_bounds(points)
    
    np.testing.assert_allclose(mins, [-1, -1, -1])
    np.testing.assert_allclose(maxs, [1, 1, 1])
    np.testing.assert_allclose(center, [0, 0, 0])
    # radius is max(norm([2, 2, 2])*0.5, 0.2) = sqrt(12)*0.5 = sqrt(3)
    np.testing.assert_allclose(radius, np.sqrt(3))

def test_quat_multiply_left():
    # Rotate 90 about Z: [cos(45), 0, 0, sin(45)]
    q_left = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
    # Identity quaternion
    q_right = np.array([[1.0, 0.0, 0.0, 0.0]])
    
    out = quat_multiply_left(q_left, q_right)
    np.testing.assert_allclose(out, [q_left])

def test_apply_scene_pose():
    params = GaussianParameters(
        means=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([[1.0]], dtype=np.float32),
        colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    # Pose: [1, 0, 0, 0, 0, pi/2] -> Translation [1,0,0], Rotation 90 about Z
    # Note: rotation convention in apply_scene_pose is [pitch, roll, yaw]
    # In src/rotations.py, yaw is euler[2], which is rotation about Z.
    pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi/2], dtype=np.float32)
    
    new_params = apply_scene_pose(params, pose)
    
    # Position: rotate [1,0,0] by 90 about Z -> [0, 1, 0], then translate [1,0,0] -> [1, 1, 0]
    np.testing.assert_allclose(new_params.means, [[1.0, 1.0, 0.0]], atol=1e-6)
    
    # Rotation: identity rotated by 90 about Z -> [cos(45), 0, 0, sin(45)]
    expected_q = np.array([[np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]], dtype=np.float32)
    np.testing.assert_allclose(new_params.rotations, expected_q, atol=1e-6)

def test_pose6_from_T():
    # Identity matrix
    T = np.eye(4)
    pose = pose6_from_T(T, matrix_type="c2w")
    np.testing.assert_allclose(pose, [0, 0, 0, 0, 0, 0])
    
    # Translate and rotate
    T = np.eye(4)
    T[0, 3] = 1.0
    # 90 deg around X
    rot = R.from_euler('x', 90, degrees=True).as_matrix()
    T[:3, :3] = rot
    
    pose = pose6_from_T(T, matrix_type="c2w")
    np.testing.assert_allclose(pose[:3], [1, 0, 0])
    np.testing.assert_allclose(pose[3:], [np.pi/2, 0, 0], atol=1e-7)

def test_pose6_from_T_w2c():
    # If matrix is w2c, it should invert it first
    # Camera at [1, 0, 0] looking along world -X
    # c2w would have translation [1, 0, 0]
    # w2c has translation -R^T * t
    T_c2w = np.eye(4)
    T_c2w[0, 3] = 1.0
    T_w2c = np.linalg.inv(T_c2w)
    
    pose = pose6_from_T(T_w2c, matrix_type="w2c")
    np.testing.assert_allclose(pose[:3], [1, 0, 0])
