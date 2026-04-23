from ultralytics.trackers.bot_sort import BOTSORT
import numpy as np
from scipy.spatial.distance import cdist
import time

class DistanceAwareBOTSORT(BOTSORT):
    STATIONARY_VEL_THRESHOLD = 2.0
    SLOWING_VEL_THRESHOLD = 8.0
    MOVING_MAX_DIST = 160.0
    STATIONARY_MAX_DIST = 80.0
    SLOWING_MAX_DIST = 40.0

    def get_dists(self, tracks, detections):
        t = time.perf_counter()
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)))

        track_centers = np.array([[t.mean[0], t.mean[1]] for t in tracks])  # (N, 2)
        velocities    = np.array([[t.mean[4], t.mean[5]] for t in tracks])  # (N, 2)
        det_centers   = np.array([[d.xywh[0], d.xywh[1]] for d in detections])  # (M, 2)

        dists = cdist(track_centers, det_centers, metric="euclidean")

        speeds = np.linalg.norm(velocities, axis=1)

        is_stationary = speeds < self.STATIONARY_VEL_THRESHOLD
        is_slowing    = (speeds >= self.STATIONARY_VEL_THRESHOLD) & (speeds < self.SLOWING_VEL_THRESHOLD)
        is_moving     = speeds >= self.SLOWING_VEL_THRESHOLD

        max_dists = np.where(is_stationary, self.STATIONARY_MAX_DIST,
                    np.where(is_slowing,    self.SLOWING_MAX_DIST,
                                            self.MOVING_MAX_DIST))[:, None]
        
        max_dists = np.where(is_stationary, self.STATIONARY_MAX_DIST, self.MOVING_MAX_DIST)[:, None]
        print(f"get_dists: {(time.perf_counter() - t)*1000:.2f}ms | tracks={len(tracks)} dets={len(detections)}")
        return np.clip(dists / max_dists, 0, 1)