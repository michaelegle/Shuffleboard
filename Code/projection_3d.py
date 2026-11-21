"""3D-aware projection of stone centers onto table coordinates.

Instead of a homography (which assumes every point lies on one plane), this solves the
camera pose in each frame from keypoints with known 3D positions (dest_x, dest_y, dest_z),
then intersects each stone's viewing ray with the horizontal plane at the stone's center
height. The X/Y of that intersection is the stone's position on the table, i.e. the point
directly below the visible center. Because the pose is solved per frame, a handheld or
moving camera (including changing camera height) is handled automatically.

Units: whatever dest_x / dest_y / dest_z / stone_height are in (all must match).

Expected dataframes
    keypoints_df: frame, class_name, confidence, pred_x, pred_y     (pixels)
    dest_df:      class_name, dest_x, dest_y, dest_z                (world units)
    stones_df:    frame, pred_x, pred_y                             (pixels)
"""

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- helpers
def _best_keypoints(keypoints_df, dest_df):
    """Highest-confidence detection per class per frame, joined to its 3D destination."""
    return (
        keypoints_df.sort_values("confidence", ascending=False)
        .drop_duplicates(["frame", "class_name"], keep="first")
        .merge(dest_df[["class_name", "dest_x", "dest_y", "dest_z"]], on="class_name", how="inner")
    )


def _well_conditioned(obj):
    """>= 4 points that are not (near) collinear in 3D."""
    if len(obj) < 4:
        return False
    s = np.linalg.svd(obj - obj.mean(axis=0), compute_uv=False)
    return s[1] > 0.05 * s[0]


def _solve_pose(sub, K, dist, reproj_px):
    """Return [rvec(3), tvec(3)] for one frame, or None."""
    obj = sub[["dest_x", "dest_y", "dest_z"]].to_numpy(np.float64)
    img = sub[["pred_x", "pred_y"]].to_numpy(np.float64)
    if not _well_conditioned(obj):
        return None
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, dist,
            reprojectionError=reproj_px,   # PIXELS (unlike findHomography, which is in dest units)
            iterationsCount=200,
            confidence=0.99,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok or inliers is None:
            return None
        idx = inliers.ravel()
        if len(idx) < 4 or not _well_conditioned(obj[idx]):
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], K, dist, rvec, tvec)
    except cv2.error:
        return None

    R, _ = cv2.Rodrigues(rvec)
    if not np.all((obj[idx] @ R.T + tvec.ravel())[:, 2] > 0):  # points must be in front of camera
        return None

    # Validate the REFINED pose. With <= 5 keypoints RANSAC can't reject anything (every point is
    # an "inlier"), and a degenerate set (e.g. 4 collinear points + 1) can refine to a pose that
    # fits nothing, with the camera 1e5 units away. Reject any pose that doesn't reproduce its
    # own inliers.
    proj, _ = cv2.projectPoints(obj[idx], rvec, tvec, K, dist)
    if np.linalg.norm(proj.reshape(-1, 2) - img[idx], axis=1).max() > 2 * reproj_px:
        return None
    return np.r_[rvec.ravel(), tvec.ravel()]


# --------------------------------------------------------------------------- intrinsics
def estimate_intrinsics(keypoints_df, dest_df, image_size, max_views=60, min_conf=0.5):
    """Estimate focal length for ONE video (principal point fixed at image center, no distortion).

    image_size = (width, height) of the video frames; keypoint pixels must be in that resolution.
    Returns (K, dist, rms_reprojection_error_px). Prefer a known focal length if you have one.
    """
    w, h = image_size
    kp = _best_keypoints(keypoints_df[keypoints_df["confidence"] >= min_conf], dest_df)

    objs, imgs = [], []
    for _, sub in kp.groupby("frame"):
        obj = sub[["dest_x", "dest_y", "dest_z"]].to_numpy(np.float32)
        if _well_conditioned(obj.astype(np.float64)):
            objs.append(obj)
            imgs.append(sub[["pred_x", "pred_y"]].to_numpy(np.float32))
    if not objs:
        raise ValueError("No frames with enough well-conditioned keypoints to calibrate.")

    pick = np.linspace(0, len(objs) - 1, min(max_views, len(objs))).astype(int)  # spread over video
    objs, imgs = [objs[i] for i in pick], [imgs[i] for i in pick]

    f0 = float(max(w, h))
    K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], dtype=np.float64)
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    )
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, (w, h), K0, np.zeros(5), flags=flags)
    return K, dist, rms


def estimate_focal_length(
    keypoints_df, dest_df, image_size, min_conf=0.5, min_kp=7, max_frames=150, f_range=(0.4, 4.0)
):
    """Robust focal-length search for ONE video (principal point at image center, no distortion).

    For each candidate focal length, solve the pose in a sample of frames and measure the
    reprojection error of the keypoints; return the focal length with the lowest MEDIAN error.
    The median ignores frames with bad detections, which calibrateCamera's least squares does not
    (it can land on a wrong focal length with a large rms, e.g. 27 px).

    Returns (K, median_rms_px). A good fit is ~1-2 px. If it is much higher, the keypoints
    don't match dest_df (wrong layout, units or labels) and no focal length will fix that.
    """
    w, h = image_size
    kp = _best_keypoints(keypoints_df[keypoints_df["confidence"] >= min_conf], dest_df)
    frames = []
    for _, sub in kp.groupby("frame"):
        obj = sub[["dest_x", "dest_y", "dest_z"]].to_numpy(np.float64)
        if len(sub) >= min_kp and _well_conditioned(obj):
            frames.append((obj, sub[["pred_x", "pred_y"]].to_numpy(np.float64)))
    if not frames:
        raise ValueError("No frames with enough well-conditioned keypoints.")
    frames = [frames[i] for i in np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)]

    def cost(f):
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        errs = []
        for obj, img in frames:
            try:
                ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
                if not ok:
                    continue
                rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, rv, tv)
                R, _ = cv2.Rodrigues(rv)
                if np.any((obj @ R.T + tv.ravel())[:, 2] <= 0):
                    continue
                proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
                errs.append(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(axis=1).mean()))
            except cv2.error:
                continue
        return np.median(errs) if len(errs) >= len(frames) / 2 else np.inf

    base = float(max(w, h))
    grid = base * np.geomspace(*f_range, 40)                 # coarse log-spaced scan
    best = int(np.argmin([cost(f) for f in grid]))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 25)                           # refine around the best cell
    costs = [cost(f) for f in fine]
    f_best = float(fine[int(np.argmin(costs))])
    K = np.array([[f_best, 0, w / 2], [0, f_best, h / 2], [0, 0, 1]], dtype=np.float64)
    return K, float(min(costs))


# --------------------------------------------------------------------------- main
def apply_3d_adjustment(
    stones_df,
    keypoints_df,
    dest_df,
    K,
    dist=None,
    stone_height=0.0,     # height of the detected stone point ABOVE the playing surface
    table_z=0.0,          # z of the playing surface in the dest_z coordinate system
    reproj_px=4.0,        # RANSAC inlier threshold for keypoints, in pixels
    max_fill_frames=30,   # don't borrow a pose from further away than this
    out_cols=("x_adj", "y_adj"),
):
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    K = np.asarray(K, float)

    # 1. Pose per frame
    kp = _best_keypoints(keypoints_df, dest_df)
    poses = {}
    for f, sub in kp.groupby("frame"):
        p = _solve_pose(sub, K, dist, reproj_px)
        if p is not None:
            poses[int(f)] = p
    if not poses:
        raise ValueError("No frame had enough valid keypoints to solve a pose.")

    stones = stones_df.copy()
    frames = stones["frame"].to_numpy(int)

    # 2. Contiguous frame axis, fill gaps in frame order (capped)
    lo = min(frames.min(), min(poses))
    hi = max(frames.max(), max(poses))
    pose_arr = (
        pd.DataFrame.from_dict(poses, orient="index")
        .reindex(np.arange(lo, hi + 1))
        .ffill(limit=max_fill_frames)
        .bfill(limit=max_fill_frames)
        .to_numpy(float)
    )  # (m, 6), NaN where no pose is close enough

    # 3. Per-frame rotation matrices and camera centers
    m = len(pose_arr)
    Rs = np.full((m, 3, 3), np.nan)
    for i in np.flatnonzero(~np.isnan(pose_arr).any(axis=1)):
        Rs[i] = cv2.Rodrigues(pose_arr[i, :3])[0]
    Cs = -np.einsum("mji,mj->mi", Rs, pose_arr[:, 3:])  # C = -R^T t

    R_rows, C_rows = Rs[frames - lo], Cs[frames - lo]

    # 4. Viewing rays through the stone centers (world frame)
    pts = stones[["pred_x", "pred_y"]].to_numpy(np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)       # normalized camera coords
    ray_cam = np.column_stack([norm, np.ones(len(norm))])
    ray_w = np.einsum("nji,nj->ni", R_rows, ray_cam)              # R^T @ ray

    # 5. Intersect with the horizontal plane at the stone's center height
    z_plane = table_z + stone_height
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (z_plane - C_rows[:, 2]) / ray_w[:, 2]
    s = np.where(s > 0, s, np.nan)                                # plane must be in front of camera
    P = C_rows + s[:, None] * ray_w

    stones[out_cols[0]], stones[out_cols[1]] = P[:, 0], P[:, 1]
    stones["pose_filled"] = ~stones["frame"].isin(poses.keys())   # borrowed from another frame
    return stones


# --------------------------------------------------------------------------- usage
if __name__ == "__main__":
    # dest_df = DEST_PTS.rename(...)  with an added dest_z column
    # K, dist, rms = estimate_intrinsics(keypoints_df, dest_df, (width, height))   # once per video
    # stones = apply_3d_adjustment(stones_df, keypoints_df, dest_df, K, dist,
    #                              stone_height=0.4, table_z=0.0)
    pass