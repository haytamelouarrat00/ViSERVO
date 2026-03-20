import os
import cv2
import numpy as np
from pathlib import Path

from src.utils_colmap import get_camera_intrinsics_and_resolution, get_camera_pose
from src.camera import VirtualCamera, create_camera_from_K
from src.servoing.fbvs_gaussian import fbvs_gaussian
from src.servoing.gaussian_fbvs import FBVS_3DGS
from src.render_pipeline import render_gaussian_view
###################################################################################################################
# LOAD CAMERA INTRINSICS FROM CAMERA.BIN
###################################################################################################################
def parse_colmap_cameras(filepath) -> dict:
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            model = parts[1]
            w, h = int(parts[2]), int(parts[3])
            f_len, cx, cy = float(parts[4]), float(parts[5]), float(parts[6])
            K = [[f_len, 0, cx], [0, f_len, cy], [0, 0, 1]]
            return {'model': model, 'width': w, 'height': h, 'K': K}

###################################################################################################################
# LOAD CAMERA POSE FROM IMAGES.TXT  
###################################################################################################################
def get_image_and_pose(images_txt, images_dir, target_id) -> dict:
    with open(images_txt, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            img_id = int(parts[0])
            if img_id == target_id:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                img_name = parts[9]
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                ])
                t = np.array([tx, ty, tz])
                image = cv2.imread(os.path.join(images_dir, img_name))
                return {'image': image, 'R': R, 't': t, 'image_name': img_name}
            next(f, None)
    return None

###################################################################################################################
#CONVERT POSE MATRIX TO 6DoF VECTOR:
###################################################################################################################
def pose_matrix_to_6dof_vector(R, t):
    from scipy.spatial.transform import Rotation
    euler = Rotation.from_matrix(R).as_euler('xyz')
    pose = np.concatenate((t.flatten(), euler))
    return pose

###################################################################################################################
#Resize Image
###################################################################################################################
def resize_image(image, scale_factor):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

###################################################################################################################
#Visualise
###################################################################################################################
def viz(img1, img2):
    img1 = resize_image(img1, 1/4)
    img2 = resize_image(img2, 1/4)
    h = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    cv2.imshow('side by side', np.hstack([img1, img2]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_data = parse_colmap_cameras(r'/home/haytam-elourrat/VS/data/garden/garden_sparse/0/cameras.txt')
    camera = create_camera_from_K(K = camera_data['K'], 
                                width = camera_data['width'], 
                                height = camera_data['height'])
    image_and_pose = get_image_and_pose(r'/home/haytam-elourrat/VS/data/garden/garden_sparse/0/images.txt', r'/home/haytam-elourrat/VS/data/garden/images', 1)
    image = image_and_pose['image']
    R, t = image_and_pose['R'], image_and_pose['t']
    pose = pose_matrix_to_6dof_vector(R.T, -R.T @ t)
    # image = resize_image(image, 1/4)
    # run_metrics_gaussian = fbvs_gaussian(
    #     scene="data/garden/garden_gs.ply",
    #     initial_pose=pose,
    #     desired_view=image,
    #     desired_pose=pose,
    #     error_tolerance=0.035,
    #     max_iters=20,
    #     verbose=True,
    #     save_frames=True,
    #     frames_dir="data/garden/debug_frames",
    #     save_video=True,
    #     camera=camera,
    #     cam_w=camera_data['width'],
    #     cam_h=camera_data['height'],
    # )
    FBVS_3DGS(
        scene="data/garden/garden_gs.ply",
        initial_pose=pose,
        desired_view=image,
        camera=camera,
    )
    