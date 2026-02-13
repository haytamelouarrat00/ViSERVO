import cv2
import numpy as np


class Features:
    def __init__(self, reference, query) -> None:
        self.reference = reference
        self.query = query

    def match_sift(self):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.reference, None)
        kp2, des2 = sift.detectAndCompute(self.query, None)
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        if len(matches) < 4:
            raise ValueError("Not enough matches found.")
        return kp1, kp2, [m for m, n in matches if m.distance < 0.7 * n.distance]

    def ransac_filter(self, kp1, kp2, matches):
        if len(matches) < 4:
            raise ValueError("Not enough matches to compute homography.")
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        if len(np.unique(src_pts, axis=0)) < 4 or len(np.unique(dst_pts, axis=0)) < 4:
            if np.allclose(src_pts, dst_pts, atol=1e-3):
                return [1] * len(matches)
            raise ValueError("Not enough unique matches to compute homography.")

        M, mask = cv2.findHomography(
            src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2), cv2.RANSAC, 5.0
        )

        if M is None or mask is None:
            if np.allclose(src_pts, dst_pts, atol=1e-3):
                return [1] * len(matches)
            raise ValueError("Homography estimation failed.")

        inliers = mask.ravel().astype(int).tolist()
        if not any(inliers) and np.allclose(src_pts, dst_pts, atol=1e-3):
            return [1] * len(matches)
        return inliers

    def extract_quad_features(self, kp1, kp2, inlier_matches):
        """
        Select 4 corner correspondences from the max-area quad and return ready arrays.

        Returns:
            rendered_features: shape (4, 2), float32
            real_features: shape (4, 2), float32
            quad_matches: list of 4 cv2.DMatch
        """
        if len(inlier_matches) == 0:
            raise ValueError("No inlier matches available to extract quad features.")

        pts_ref = [kp1[m.queryIdx].pt for m in inlier_matches]
        quad_ref = Features.max_area_quad(pts_ref)
        if quad_ref is None:
            raise ValueError("Could not compute max-area quad from inlier matches.")

        quad_pts = np.asarray(quad_ref, dtype=np.float32).reshape(-1, 2)
        ref_pts = np.asarray(
            [kp1[m.queryIdx].pt for m in inlier_matches], dtype=np.float32
        )

        selected = []
        used_indices = set()
        for q in quad_pts:
            sq_dist = np.sum((ref_pts - q[None, :]) ** 2, axis=1)
            for idx in np.argsort(sq_dist):
                idx_int = int(idx)
                if idx_int not in used_indices:
                    used_indices.add(idx_int)
                    selected.append(inlier_matches[idx_int])
                    break

        if len(selected) != 4:
            raise ValueError(
                "Could not select exactly 4 matches from max-area quad corners."
            )

        rendered_features = np.asarray(
            [kp1[m.queryIdx].pt for m in selected], dtype=np.float32
        )
        real_features = np.asarray(
            [kp2[m.trainIdx].pt for m in selected], dtype=np.float32
        )
        return rendered_features, real_features, selected

    @staticmethod
    def triangle_area(a, b, c):
        return (
            abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2
        )

    @staticmethod
    def max_area_quad(points):
        hull = cv2.convexHull(np.array(points, dtype=np.float32)).squeeze()
        h = len(hull)

        if h < 4:
            return None

        max_area = 0
        best_quad = None

        for i in range(h):
            for j in range(i + 2, h):
                if (j + 1) % h == i:
                    continue  # adjacent edges, skip

                # Find k maximizing area(i, k, j)
                k = (i + 1) % h
                best_k = k
                while True:
                    next_k = (k + 1) % h
                    if Features.triangle_area(
                        hull[i], hull[next_k], hull[j]
                    ) > Features.triangle_area(hull[i], hull[k], hull[j]):
                        k = next_k
                        best_k = k
                    else:
                        break

                # Find l maximizing area(i, j, l)
                l = (j + 1) % h
                best_l = l
                while True:
                    next_l = (l + 1) % h
                    if Features.triangle_area(
                        hull[i], hull[j], hull[next_l]
                    ) > Features.triangle_area(hull[i], hull[j], hull[l]):
                        l = next_l
                        best_l = l
                    else:
                        break

                curr_area = Features.triangle_area(
                    hull[i], hull[best_k], hull[j]
                ) + Features.triangle_area(hull[i], hull[j], hull[best_l])

                if curr_area > max_area:
                    max_area = curr_area
                    best_quad = (hull[i], hull[best_k], hull[j], hull[best_l])

        return best_quad
