import math
import cv2
import numpy as np

class HybridStoreLocalizer:
    """
    Implements Section V-D: On-device IMU Pedestrian Dead Reckoning (PDR)
    with periodic RANSAC-PnP 6-DoF visual pose drift reset.
    """
    def __init__(self, x0: float = 0.0, y0: float = 0.0, theta0: float = 0.0):
        self.x = x0
        self.y = y0
        self.theta = theta0

    def update_pdr(self, step_length: float, delta_heading_rad: float):
        self.theta += delta_heading_rad
        self.x += step_length * math.cos(self.theta)
        self.y += step_length * math.sin(self.theta)
        return self.x, self.y, self.theta

    def reset_drift_pnp(self, object_pts_3d: np.ndarray, image_pts_2d: np.ndarray, camera_matrix: np.ndarray):
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            object_pts_3d.astype(np.float32),
            image_pts_2d.astype(np.float32),
            camera_matrix.astype(np.float32),
            None
        )
        if success:
            self.x = float(tvec[0][0])
            self.y = float(tvec[2][0])
        return self.x, self.y, self.theta