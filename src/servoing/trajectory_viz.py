from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _AVAILABLE = True
except Exception:
    _AVAILABLE = False

# Length of the camera-frame axes drawn at the current position (in scene units)
_AXIS_LEN = 0.05


def _rotation_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix from XYZ Euler angles (radians)."""
    rx, ry, rz = float(euler[0]), float(euler[1]), float(euler[2])

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


class TrajectoryVisualizer:
    """
    Real-time 3D camera trajectory viewer.

    Draws:
    - Ground-truth waypoints as a dashed orange line (set once before the loop).
    - The followed path as a solid blue line updated each step.
    - A camera-frame triad (R=X, G=Y, B=Z) at the latest pose showing where
      the camera is pointing.

    No scatter markers are drawn for intermediate or final poses.
    """

    def __init__(self, title: str = "Camera Trajectory") -> None:
        self._poses: list[np.ndarray] = []
        self._gt_poses: list[np.ndarray] = []
        self._active = False

        if not _AVAILABLE:
            return
        try:
            plt.ion()
            self.fig = plt.figure(figsize=(9, 7))
            self.ax = self.fig.add_subplot(111, projection="3d")
            self.fig.suptitle(title, fontsize=12)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            plt.show(block=False)
            plt.pause(0.001)
            self._active = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    def set_ground_truth(self, poses: list[np.ndarray]) -> None:
        """
        Supply the full ground-truth waypoint sequence before servoing starts.
        Drawn as a dashed orange line; remains visible throughout execution.
        """
        if not self._active:
            return
        self._gt_poses = [np.asarray(p, dtype=np.float64).copy() for p in poses]
        self._redraw()

    # ------------------------------------------------------------------
    def add_pose(self, pose: np.ndarray, label: str = "") -> None:
        """Append a servo pose and refresh the 3D view immediately."""
        if not self._active:
            return
        self._poses.append(np.asarray(pose, dtype=np.float64).copy())
        self._redraw()

    # ------------------------------------------------------------------
    def _draw_camera_frame(self, pose: np.ndarray) -> None:
        """
        Draw X (red), Y (green), Z (blue) axes of the camera frame at `pose`.

        `pose` is a 6-vector [x, y, z, rx, ry, rz] where r* are XYZ Euler
        angles (radians) that define the camera-to-world rotation.
        """
        origin = pose[:3]
        R = _rotation_from_euler_xyz(pose[3:6])

        # Columns of R are the camera X, Y, Z axes expressed in world coords
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        scale = self._axis_scale()

        for axis, color in zip([x_axis, y_axis, z_axis], ["red", "green", "blue"]):
            end = origin + axis * scale
            self.ax.quiver(
                origin[0], origin[1], origin[2],
                end[0] - origin[0], end[1] - origin[1], end[2] - origin[2],
                color=color, linewidth=2.0, arrow_length_ratio=0.3,
                zorder=10,
            )

    # ------------------------------------------------------------------
    def _axis_scale(self) -> float:
        """
        Choose axis length as a fraction of the scene bounding box so it
        stays readable regardless of the scene scale.
        """
        all_pts = []
        if self._gt_poses:
            all_pts.extend(self._gt_poses)
        if self._poses:
            all_pts.extend(self._poses)

        if len(all_pts) < 2:
            return _AXIS_LEN

        pts = np.array(all_pts)[:, :3]
        span = np.max(pts, axis=0) - np.min(pts, axis=0)
        diag = float(np.linalg.norm(span))
        return max(diag * 0.06, _AXIS_LEN)

    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        try:
            self.ax.cla()

            # ── Ground-truth trajectory ──────────────────────────────────
            if self._gt_poses:
                gt = np.array(self._gt_poses)
                gt_pos = gt[:, :3]
                if len(gt_pos) > 1:
                    self.ax.plot(
                        gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                        color="orange", linewidth=1.5, linestyle="--",
                        alpha=0.7, label="ground truth", zorder=3,
                    )

            # ── Followed (estimated) trajectory ─────────────────────────
            if self._poses:
                poses = np.array(self._poses)
                positions = poses[:, :3]

                if len(positions) > 1:
                    self.ax.plot(
                        positions[:, 0], positions[:, 1], positions[:, 2],
                        color="steelblue", linewidth=1.8, alpha=0.9,
                        label="followed", zorder=5,
                    )

                # Camera-frame triad at the latest pose
                self._draw_camera_frame(self._poses[-1])

            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.legend(fontsize=8)

            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            self._active = False

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the current figure to disk."""
        if not self._active:
            return
        try:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")
        except Exception:
            pass

    def keep_open(self) -> None:
        """Block until the user closes the window (call at end of script)."""
        if not self._active:
            return
        try:
            plt.ioff()
            plt.show(block=True)
        except Exception:
            pass

    def close(self) -> None:
        if not self._active:
            return
        try:
            plt.close(self.fig)
        except Exception:
            pass
        self._active = False
