import numpy as np


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    w, x, y, z = q

    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles [pitch, roll, yaw] (radians) to quaternion [w, x, y, z].
    """
    pitch, roll, yaw = np.asarray(euler, dtype=np.float64).reshape(3)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    quat = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm
