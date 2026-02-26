import numpy as np
import pytest
from src.control import (
    geometric_error,
    ZNSSD,
    normalize_features,
    interaction_matrix,
    velocity,
    photometric_error,
    luminance_interaction_matrix,
    image_gradient,
    PhotometricVisualServoingZN
)
from src.camera import VirtualCamera, CameraIntrinsics

@pytest.fixture
def default_camera():
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0, 1.0]])
    intrinsics = CameraIntrinsics(width=640, height=480, K=K)
    return VirtualCamera(intrinsics)

# ===========================================================================
# geometric_error
# ===========================================================================

def test_geometric_error_zero():
    f = np.array([[100.0, 200.0], [300.0, 400.0]])
    err, mean_err = geometric_error(f, f)
    np.testing.assert_allclose(err, 0.0)
    assert mean_err == 0.0

def test_geometric_error_nonzero():
    f_des = np.array([[100.0, 200.0]])
    f_curr = np.array([[105.0, 202.0]])
    err, mean_err = geometric_error(f_des, f_curr)
    # err = current - desired = [5, 2]
    np.testing.assert_allclose(err.flatten(), [5.0, 2.0])
    assert np.isclose(mean_err, np.sqrt(5**2 + 2**2))

def test_geometric_error_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        geometric_error(np.zeros((2, 2)), np.zeros((3, 2)))

# ===========================================================================
# ZNSSD
# ===========================================================================

def test_znssd_identical():
    # When images are identical, ZNSSD should be 0
    f = np.random.rand(10, 10)
    score = ZNSSD(f, f)
    assert np.isclose(score, 0.0, atol=1e-10)

def test_znssd_robust_to_offset():
    # ZNSSD(I, I + b) should be 0 based on current implementation
    f1 = np.random.rand(10, 10)
    f2 = f1 + 0.5
    score = ZNSSD(f1, f2)
    assert np.isclose(score, 0.0, atol=1e-10)

def test_znssd_not_robust_to_scale():
    # The current implementation of ZNSSD in control.py is NOT robust to scale
    # ZNSSD(I, a*I) = (a-1)^2 / a
    f1 = np.random.rand(10, 10)
    a = 2.0
    f2 = a * f1
    score = ZNSSD(f1, f2)
    expected = (a - 1)**2 / a
    assert np.isclose(score, expected, atol=1e-10)

def test_znssd_different():
    f1 = np.array([1, 2, 3, 4], dtype=float)
    f2 = np.array([4, 3, 2, 1], dtype=float)
    score = ZNSSD(f1, f2)
    assert score > 0.0

# ===========================================================================
# normalize_features
# ===========================================================================

def test_normalize_features(default_camera):
    # Principal point [320, 240] should normalize to [0, 0]
    pixels = np.array([[320.0, 240.0]])
    normalized = normalize_features(pixels, default_camera)
    np.testing.assert_allclose(normalized, [[0.0, 0.0]], atol=1e-7)

    # Point [320 + 800, 240] should normalize to [1, 0]
    pixels = np.array([[1120.0, 240.0]])
    normalized = normalize_features(pixels, default_camera)
    np.testing.assert_allclose(normalized, [[1.0, 0.0]], atol=1e-7)

def test_normalize_features_invalid_shape(default_camera):
    with pytest.raises(ValueError, match="Expected features to have shape"):
        normalize_features(np.zeros(2), default_camera)

# ===========================================================================
# interaction_matrix
# ===========================================================================

def test_interaction_matrix_shape():
    features = np.array([[0.1, 0.2], [-0.1, -0.2]])
    L = interaction_matrix(features, depths=2.0)
    assert L.shape == (4, 6)

def test_interaction_matrix_values():
    # Point at (0, 0) in normalized coords with Z=1
    # L = [-1/Z, 0, x/Z, xy, -(1+x^2), y]
    #     [0, -1/Z, y/Z, 1+y^2, -xy, -x]
    # For (0,0), Z=1:
    # L = [-1, 0, 0, 0, -1, 0]
    #     [0, -1, 0, 1, 0, 0]
    features = np.array([[0.0, 0.0]])
    L = interaction_matrix(features, depths=1.0)
    expected = np.array([
        [-1.0,  0.0, 0.0, 0.0, -1.0, 0.0],
        [ 0.0, -1.0, 0.0, 1.0,  0.0, 0.0]
    ])
    np.testing.assert_allclose(L, expected, atol=1e-7)

def test_interaction_matrix_invalid_depth():
    with pytest.raises(ValueError, match="must be positive"):
        interaction_matrix(np.zeros((1, 2)), depths=0.0)

# ===========================================================================
# velocity
# ===========================================================================

def test_velocity_calculation():
    # L = identity-like for 3 points to make it square and invertible for simplicity
    # but L is 2N x 6. Let's use 3 points -> 6x6.
    L = np.eye(6)
    error = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # v = -gain * L_pinv @ error = -1.0 * I @ [1, 0, 0, 0, 0, 0] = [-1, 0, 0, 0, 0, 0]
    v = velocity(L, error, gain=1.0)
    np.testing.assert_allclose(v, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# ===========================================================================
# photometric_error
# ===========================================================================

def test_photometric_error():
    img1 = np.ones((10, 10)) * 100
    img2 = np.ones((10, 10)) * 110
    err, cost = photometric_error(img1, img2)
    # err = current - desired = 110 - 100 = 10
    # cost = 0.5 * sum(err^2) = 0.5 * 100 * 100 = 5000
    assert err.shape == (100,)
    np.testing.assert_allclose(err, 10.0)
    assert np.isclose(cost, 5000.0)

# ===========================================================================
# image_gradient
# ===========================================================================

def test_image_gradient():
    # Create a ramp image
    img = np.tile(np.arange(10), (10, 1)).astype(np.float32) # gradient in X
    Ix, Iy = image_gradient(img)
    # Ix should be positive, Iy should be near zero
    assert np.mean(Ix) > 0
    assert np.allclose(Iy, 0.0, atol=1e-5)

# ===========================================================================
# PhotometricVisualServoingZN
# ===========================================================================

class TestPhotometricVisualServoingZN:
    def test_zero_mean_normalize(self, default_camera):
        pvs = PhotometricVisualServoingZN(default_camera)
        img = np.array([[10, 20], [30, 40]], dtype=float)
        normalized = pvs.zero_mean_normalize(img)
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
        # mean is 25. normalized should be [[-15, -5], [5, 15]]
        np.testing.assert_allclose(normalized, [[-15, -5], [5, 15]])

    def test_compute_control_velocity(self, default_camera):
        pvs = PhotometricVisualServoingZN(default_camera, lambda_gain=1.0)
        h, w = 20, 20
        I_des = np.zeros((h, w), dtype=np.float32)
        I_des[5:15, 5:15] = 255
        
        I_curr = np.zeros((h, w), dtype=np.float32)
        I_curr[6:16, 5:15] = 255 # Shifted by 1 pixel in Y
        
        Ix, Iy = image_gradient(I_curr)
        Z = 2.0
        
        v, cost = pvs.compute_control_velocity(I_curr, I_des, Ix, Iy, Z)
        
        assert v.shape == (6,)
        assert cost > 0
        assert len(pvs.cost_history) == 1
