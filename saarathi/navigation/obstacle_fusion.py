class MultiSensorObstacleFusion:
    """
    Implements Section V-E (Eq. 26):
    Combines RGB-D depth, monocular triangle similarity d_tri = (W_real * f_cam) / w_pix,
    and ToF/ultrasonic rangefinding to trigger warnings when d_obs < tau_obs.
    """
    def __init__(self, tau_obs: float = 1.2, f_cam: float = 600.0):
        self.tau_obs = tau_obs
        self.f_cam = f_cam

    def triangle_similarity_distance(self, w_real_meters: float, w_pix: float) -> float:
        return (w_real_meters * self.f_cam) / max(w_pix, 1.0)

    def estimate_min_obstacle_distance(self, rgbd_depth_m: float, tof_dist_m: float,
                                       w_real_m: float = 0.5, bbox_width_px: float = 200.0):
        d_tri = self.triangle_similarity_distance(w_real_m, bbox_width_px)
        # Conservative safety fusion takes minimum valid distance across sensors
        d_obs = min(rgbd_depth_m, tof_dist_m, d_tri)
        is_hazard = d_obs < self.tau_obs
        return d_obs, is_hazard