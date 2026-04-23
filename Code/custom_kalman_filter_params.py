from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
import numpy as np

class CollisionAwareKalmanFilter(KalmanFilterXYWH):
    # Velocity threshold below which a stone is considered stationary (pixels/frame)
    STATIONARY_VEL_THRESHOLD = 2.0
    # Extra uncertainty multiplier injected for stationary stones
    STATIONARY_NOISE_MULTIPLIER = 8.0

    def initiate(self, measurement):
        mean, covariance = super().initiate(measurement)
        covariance *= 4.0
        return mean, covariance

    def predict(self, mean, covariance):
        mean, covariance = super().predict(mean, covariance)
        covariance[4:, 4:] *= 2.0

        # Check if stone is stationary (vx, vy are indices 4 and 5)
        vx, vy = mean[4], mean[5]
        speed = np.sqrt(vx**2 + vy**2)

        if speed < self.STATIONARY_VEL_THRESHOLD:
            # Stone is still — widen positional uncertainty so a collision hit
            # doesn't cause the filter to reject the new measurement
            covariance[:4, :4] *= self.STATIONARY_NOISE_MULTIPLIER

        return mean, covariance