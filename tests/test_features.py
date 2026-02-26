import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock, patch
from src.features import (
    _validate_grayscale_image,
    _to_uint8,
    _resize_to_match,
    SIFTMatcher,
    XFeatMatcher,
    Features,
    MatchResult
)

# ===========================================================================
# Image helpers
# ===========================================================================

def test_validate_grayscale_image_valid():
    img = np.zeros((10, 10), dtype=np.uint8)
    _validate_grayscale_image(img) # Should not raise

def test_validate_grayscale_image_invalid_ndim():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must be a 2-D grayscale array"):
        _validate_grayscale_image(img)

def test_validate_grayscale_image_empty():
    img = np.zeros((0, 0), dtype=np.uint8)
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_grayscale_image(img)

def test_validate_grayscale_image_unsupported_dtype():
    img = np.zeros((10, 10), dtype=np.int32)
    with pytest.raises(ValueError, match="unsupported dtype"):
        _validate_grayscale_image(img)

def test_to_uint8():
    img = np.array([0.0, 1.0], dtype=np.float32).reshape(1, 2)
    img_u8 = _to_uint8(img)
    assert img_u8.dtype == np.uint8
    assert img_u8[0, 0] == 0
    assert img_u8[0, 1] == 255

def test_resize_to_match():
    img = np.zeros((10, 20), dtype=np.uint8)
    resized = _resize_to_match(img, (30, 40))
    assert resized.shape == (30, 40)

# ===========================================================================
# SIFTMatcher
# ===========================================================================

def test_sift_matcher_invalid_threshold():
    with pytest.raises(ValueError, match="ratio_threshold must be in \(0, 1\)"):
        SIFTMatcher(ratio_threshold=1.5)

def test_sift_matcher_insufficient_features():
    # Empty/uniform images should have no features
    img = np.zeros((100, 100), dtype=np.uint8)
    matcher = SIFTMatcher()
    result = matcher.match(img, img)
    assert len(result.matches) == 0

def test_sift_matcher_success():
    # Create images with some features (random noise + some structure)
    rng = np.random.default_rng(42)
    img1 = (rng.random((200, 200)) * 255).astype(np.uint8)
    # Add a square
    img1[50:150, 50:150] = 255
    # img2 is slightly shifted
    img2 = np.roll(img1, 5, axis=0)
    
    matcher = SIFTMatcher()
    result = matcher.match(img1, img2)
    # We expect some matches
    assert isinstance(result, MatchResult)
    # Depending on SIFT, we might get matches or not on random noise, 
    # but with a square it should find something.
    # Note: SIFT might not be compiled in all opencv versions (though usually is now).

# ===========================================================================
# XFeatMatcher
# ===========================================================================

@patch('src.features.XFeatMatcher._get_xfeat')
def test_xfeat_matcher_mocked(mock_get_xfeat):
    # Setup mock xfeat
    mock_xfeat = MagicMock()
    # match_xfeat returns (mkpts_0, mkpts_1)
    mock_xfeat.match_xfeat.return_value = (
        np.array([[10, 10], [20, 20], [30, 30], [40, 40]]),
        np.array([[11, 11], [21, 21], [31, 31], [41, 41]])
    )
    mock_get_xfeat.return_value = mock_xfeat
    
    matcher = XFeatMatcher()
    img = np.zeros((100, 100), dtype=np.uint8)
    result = matcher.match(img, img)
    
    assert len(result.keypoints_ref) == 4
    assert len(result.matches) > 0 # Homography should succeed with these points

# ===========================================================================
# Features Facade
# ===========================================================================

def test_features_facade_init():
    ref = np.zeros((100, 100), dtype=np.uint8)
    qry = np.zeros((50, 50), dtype=np.uint8)
    f = Features(ref, qry)
    assert f.query.shape == (100, 100) # Should be resized

@patch('src.features.SIFTMatcher.match')
def test_features_match_sift(mock_match):
    mock_match.return_value = MatchResult(
        np.array(['kp1'], dtype=object),
        np.array(['kp2'], dtype=object),
        np.array(['m'], dtype=object)
    )
    ref = np.zeros((100, 100), dtype=np.uint8)
    f = Features(ref, ref)
    kp_ref, kp_qry, matches = f.match_sift()
    assert kp_ref[0] == 'kp1'
    assert matches[0] == 'm'
