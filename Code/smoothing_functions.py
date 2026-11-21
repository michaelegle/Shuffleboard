"""Modular smoothing for stone tracks in table coordinates.

A *smoother* is any function (n, 2) array -> (n, 2) array. `smooth_dataframe` applies one
to every contiguous track in a dataframe, grouped by whatever columns you choose, so the
same driver serves both passes:

    Pass 1 (no toss IDs yet):  group_cols=["stone_id"],             smoother=gaussian_smoother(1.5)
    Pass 2 (toss IDs exist):   group_cols=["toss_id", "stone_id"],  smoother=kalman_smoother(kf)

Units: cm for position, frames for time. Thresholds named *_cm_s are per second (use fps).
"""

from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

Smoother = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- driver
def iter_segments(
    df: pd.DataFrame, group_cols: Sequence[str], frame_col: str = "frame", min_len: int = 5
) -> Iterator[pd.DataFrame]:
    """Yield contiguous (no frame gaps), frame-sorted segments within each group."""
    for _, g in df.sort_values([*group_cols, frame_col]).groupby(list(group_cols), sort=False):
        seg_id = (g[frame_col].diff() != 1).cumsum()
        for _, seg in g.groupby(seg_id, sort=False):
            if len(seg) >= min_len:
                yield seg


def smooth_dataframe(
    df: pd.DataFrame,
    smoother: Smoother,
    group_cols: Sequence[str],
    xcol: str = "X",
    ycol: str = "Y",
    out_cols: Sequence[str] = ("X_s", "Y_s"),
    frame_col: str = "frame",
    min_len: int = 5,
) -> pd.DataFrame:
    """Apply `smoother` to each contiguous track. Segments shorter than min_len pass through."""
    assert df.index.is_unique, "reset_index(drop=True) first"
    df = df.copy()
    df[out_cols[0]], df[out_cols[1]] = df[xcol], df[ycol]
    for seg in iter_segments(df, group_cols, frame_col, min_len):
        df.loc[seg.index, list(out_cols)] = smoother(seg[[xcol, ycol]].to_numpy(float))
    return df


def add_speed(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    fps: float = 30,
    sigma: float = 1.5,
    xcol: str = "X",
    ycol: str = "Y",
    out_col: str = "speed",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """Smoothed speed (cm/s) via a Gaussian-derivative filter. Useful for toss detection."""
    assert df.index.is_unique, "reset_index(drop=True) first"
    df = df.copy()
    df[out_col] = 0.0
    for seg in iter_segments(df, group_cols, frame_col, min_len=3):
        v = gaussian_filter1d(seg[[xcol, ycol]].to_numpy(float), sigma, axis=0, order=1, mode="nearest")
        df.loc[seg.index, out_col] = np.hypot(v[:, 0], v[:, 1]) * fps
    return df


# --------------------------------------------------------------------------- smoothers
def gaussian_smoother(sigma: float = 1.5) -> Smoother:
    """Light, positive-weight smoothing. Good for pass 1."""
    return lambda xy: gaussian_filter1d(xy, sigma, axis=0, mode="nearest")


def kalman_smoother(kf: "CollisionAwareKF2D", fps: float = 30, pad: int = 2, **seg_kwargs) -> Smoother:
    """Rest runs -> median, moving runs -> KF + RTS. Good for pass 2."""
    return lambda xy: smooth_track(xy, kf, fps=fps, pad=pad, **seg_kwargs)


# --------------------------------------------------------------------------- Kalman core
class CollisionAwareKF2D:
    """Constant-velocity KF, state [x, y, vx, vy], with RTS backward pass.

    When the stone looks stationary, position covariance is inflated ONLY in a temporary
    copy used to compute the gain. The stored covariance follows the standard recursion,
    so it doesn't compound and the RTS pass stays valid.
    """

    def __init__(
        self,
        r=0.5,                  # measurement variance (cm^2); see estimate_r
        q_pos=0.01,
        q_vel=0.02,             # larger velocity process noise = maneuver allowance
        stationary_speed=0.2,   # cm/frame
        stationary_mult=8.0,
        speed_window=3,
        p0=40.0,
    ):
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.R = np.eye(2) * r
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.stationary_speed = stationary_speed
        self.stationary_mult = stationary_mult
        self.speed_window = speed_window
        self.p0 = p0

    def run(self, xy: np.ndarray) -> np.ndarray:
        n = len(xy)
        if n < 3:
            return xy.copy()

        I4 = np.eye(4)
        x = np.array([xy[0, 0], xy[0, 1], 0.0, 0.0])
        P = I4 * self.p0
        xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))
        xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4))
        w = self.speed_window

        for k in range(n):
            if k > 0:
                x = self.F @ x
                P = self.F @ P @ self.F.T + self.Q
            xp[k], Pp[k] = x, P

            if k - 1 - w >= 0:
                speed = np.hypot(*(xf[k - 1, :2] - xf[k - 1 - w, :2])) / w
            else:
                speed = np.hypot(x[2], x[3])

            P_gain = P.copy()
            if speed < self.stationary_speed:
                P_gain[:2, :2] *= self.stationary_mult

            S = self.H @ P_gain @ self.H.T + self.R
            K = P_gain @ self.H.T @ np.linalg.inv(S)
            x = x + K @ (xy[k] - self.H @ x)
            A = I4 - K @ self.H
            P = A @ P @ A.T + K @ self.R @ K.T  # Joseph form, valid for any gain
            xf[k], Pf[k] = x, P

        xs = xf.copy()
        for k in range(n - 2, -1, -1):
            C = Pf[k] @ self.F.T @ np.linalg.inv(Pp[k + 1])
            xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        return xs[:, :2]


# --------------------------------------------------------------------------- motion routing
def segment_motion(xy, fps=30, start_cm_s=8.0, stop_cm_s=4.0, min_rest=10, sigma=1.5) -> np.ndarray:
    """Boolean array, True where moving. Hysteresis + minimum rest length."""
    v = gaussian_filter1d(xy, sigma, axis=0, order=1, mode="nearest") * fps
    speed = np.hypot(v[:, 0], v[:, 1])

    moving = np.zeros(len(xy), bool)
    state = False
    for i, s in enumerate(speed):
        if not state and s > start_cm_s:
            state = True
        elif state and s < stop_cm_s:
            state = False
        moving[i] = state

    for s, e, is_moving in _runs(moving):
        if not is_moving and (e - s) < min_rest:
            moving[s:e] = True
    return moving


def _runs(mask):
    edges = np.flatnonzero(np.diff(mask.astype(int))) + 1
    return [(s, e, bool(mask[s])) for s, e in zip(np.r_[0, edges], np.r_[edges, len(mask)])]


def smooth_track(xy, kf, fps=30, pad=2, **seg_kwargs) -> np.ndarray:
    n = len(xy)
    out = xy.copy()
    for s, e, is_moving in _runs(segment_motion(xy, fps, **seg_kwargs)):
        if is_moving:
            s2, e2 = max(0, s - pad), min(n, e + pad)
            out[s:e] = kf.run(xy[s2:e2])[s - s2 : e - s2]
        else:
            out[s:e] = np.median(xy[s:e], axis=0)
    return out


def estimate_r(
    df, group_cols, fps=30, xcol="X", ycol="Y", frame_col="frame", min_rest_len=5, **seg_kwargs
) -> float:
    """Pooled variance of table coords over resting runs -> measurement noise `r`."""
    resid = []
    for seg in iter_segments(df, group_cols, frame_col):
        xy = seg[[xcol, ycol]].to_numpy(float)
        for s, e, is_moving in _runs(segment_motion(xy, fps, **seg_kwargs)):
            if not is_moving and e - s >= min_rest_len:
                resid.append(xy[s:e] - np.median(xy[s:e], axis=0))
    return float(np.var(np.vstack(resid))) if resid else 0.5


# --------------------------------------------------------------------------- usage
if __name__ == "__main__":
    # df: columns frame, stone_id, X, Y (table coords, cm); unique index
    #
    # Pass 1: clean speed signal for toss detection (no toss IDs yet)
    # df = smooth_dataframe(df, gaussian_smoother(1.5), group_cols=["stone_id"],
    #                       out_cols=("X_p1", "Y_p1"))
    # df = add_speed(df, group_cols=["stone_id"], xcol="X_p1", ycol="Y_p1")
    # ... assign df["toss_id"] from df["speed"] ...
    #
    # Pass 2: final trajectories, from the UNSMOOTHED X/Y, per video
    # r = estimate_r(df, group_cols=["toss_id", "stone_id"], fps=30)
    # kf = CollisionAwareKF2D(r=r)
    # df = smooth_dataframe(df, kalman_smoother(kf, fps=30), group_cols=["toss_id", "stone_id"])
    pass