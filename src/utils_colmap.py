import pycolmap
import numpy as np
import cv2
import os

from src.rotations import quat_to_matrix


def get_camera_intrinsics(colmap_model_path: str) -> np.ndarray:
    reconstruction = pycolmap.Reconstruction(colmap_model_path)
    camera = next(iter(reconstruction.cameras.values()))
    return camera.calibration_matrix()


def get_camera_intrinsics_and_resolution(
    colmap_model_path: str,
) -> tuple[np.ndarray, int, int]:
    reconstruction = pycolmap.Reconstruction(colmap_model_path)
    camera = next(iter(reconstruction.cameras.values()))
    K = camera.calibration_matrix()
    return K, int(camera.width), int(camera.height)


def qvec_to_rotmat(qvec):
    return quat_to_matrix(np.asarray(qvec, dtype=np.float64))


def get_camera_pose(colmap_model_path: str, image_path: str) -> np.ndarray:
    reconstruction = pycolmap.Reconstruction(colmap_model_path)
    image_name = os.path.basename(image_path)
    image = next(
        (img for img in reconstruction.images.values() if img.name == image_name),
        None,
    )
    if image is None:
        raise ValueError(f"Image '{image_name}' not found in COLMAP model.")

    # Get rotation and translation using cam_from_world() method
    pose = image.cam_from_world()
    R = pose.rotation.matrix()
    t = pose.translation

    # Convert to Rodrigues (axis-angle)
    rvec, _ = cv2.Rodrigues(R)
    pose_6d = np.concatenate([rvec.flatten(), t])
    return pose_6d


if __name__ == "__main__":
    print(get_camera_intrinsics("data/sparse/0"))
