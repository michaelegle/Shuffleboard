# Functions for building the data

import os
import time
from datetime import datetime
from types import MethodType
from smoothing_functions import *
from projection_3d import *

import cv2
import numpy as np
import pandas as pd
import yaml
from custom_botsort import DistanceAwareBOTSORT
from custom_kalman_filter_params import CollisionAwareKalmanFilter
from filterpy.kalman import KalmanFilter
from scipy.signal import medfilt, savgol_filter
from ultralytics import YOLO
from ultralytics.trackers import bot_sort

SOURCE = "Film/test_clip.MOV"
TRACKER = "Code/custom_botsort_params.yaml"
# TODO - finetune the keypoint tracking parameters?
KEYPOINT_TRACKER = "Code/keypoint_tracking_params.yaml"

# TODO - figure out the stupid fucking keyboard shortcuts on this god forsaken IDE


def make_kalman_filter(initial_x, initial_y, process_noise=0.01, measurement_noise=5.0):
    """
    2D Kalman filter with constant velocity model.
    State vector: [x, y, vx, vy]
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State transition matrix (position += velocity each step)
    kf.F = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )

    # Measurement matrix (we observe x and y, not velocity)
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    kf.R *= measurement_noise  # measurement noise
    kf.Q *= process_noise  # process noise

    kf.x = np.array([[initial_x], [initial_y], [0.0], [0.0]])  # initial state
    kf.P *= 10.0  # initial uncertainty

    return kf


def smooth_trajectory(group, process_noise=0.1, measurement_noise=1.0):
    """Apply Kalman filter to a single (toss, stone) trajectory."""
    group = group.sort_values("frame")
    xs, ys = group["x"].values, group["y"].values

    kf = make_kalman_filter(xs[0], ys[0], process_noise, measurement_noise)

    smoothed_x, smoothed_y = [], []
    for x, y in zip(xs, ys):
        kf.predict()
        kf.update(np.array([[x], [y]]))
        smoothed_x.append(kf.x[0, 0])
        smoothed_y.append(kf.x[1, 0])

    group = group.copy()
    group["x"] = smoothed_x
    group["y"] = smoothed_y
    return group


def predict_new_video(
    video_path,
    window,
    stone_model_dir="Models/stone_detection/model_saves/weights/best.pt",
    keypoint_model_dir="Models/keypoint_detection3/weights/best.pt",
):

    stone_model = YOLO(stone_model_dir)
    keypoint_model = YOLO(keypoint_model_dir)
    # Step 1: stream=True gives us a generator — grab just the first frame
    #         so Ultralytics builds the predictor AND registers trackers
    generator = stone_model.track(
        source=video_path,
        tracker=TRACKER,
        persist=True,
        device="mps",
        stream=True,
        save=True,
    )

    first_result = next(generator)  # just one frame to init predictor.trackers

    # Step 2: now trackers exists — patch the live instance
    tracker = stone_model.predictor.trackers[0]
    tracker.__class__ = DistanceAwareBOTSORT
    tracker.kalman_filter.__class__ = CollisionAwareKalmanFilter
    tracker.reset()  # wipe state from the dummy frame

    # Step 3: close the generator, run fresh on the full video
    generator.close()

    # Step 4: full inference with patches active
    stone_results = stone_model.track(
        source=video_path,
        tracker=TRACKER,
        iou = 0.3, # keep low for collisions
        conf = 0.1,
        save=True,
        save_json=False,
        persist=True,
        device="mps",
        stream=True,
    )

    keypoint_results = keypoint_model.predict(
        source=video_path,
        save=True,
        conf=0.25,
        save_json=True,
        device="mps",
        stream=True,
    )

    all_stones_df = pd.DataFrame()
    for frame_idx, result in enumerate(stone_results):
        frame_predictions = pd.DataFrame()
        for box in result.boxes:
            prediction = {
                "frame": frame_idx,
                "class_id": int(box.cls),
                "class_name": stone_model.names[int(box.cls)],
                "confidence": float(box.conf),
                "track_id": int(box.id) if box.id is not None else None,
                "pred_x": float(box.xywh[0][0]),
                "pred_y": float(box.xywh[0][1]),
            }
            # TODO - find issue from frames 400-600 being missed for some reason
            prediction_df = pd.DataFrame(prediction, index=[0])
            frame_predictions = pd.concat([frame_predictions, prediction_df])
        all_stones_df = pd.concat([all_stones_df, frame_predictions])

    all_keypoints_df = pd.DataFrame()
    for frame_idx, result in enumerate(keypoint_results):
        frame_predictions = pd.DataFrame()
        for box in result.boxes:
            prediction = {
                "frame": frame_idx,
                "class_id": int(box.cls),
                "class_name": keypoint_model.names[int(box.cls)],
                "confidence": float(box.conf),
                "pred_x": float(box.xywh[0][0]),
                "pred_y": float(box.xywh[0][1]),
            }
            prediction_df = pd.DataFrame(prediction, index=[0])
            frame_predictions = pd.concat([frame_predictions, prediction_df])
        all_keypoints_df = pd.concat([all_keypoints_df, frame_predictions])

    return all_stones_df, all_keypoints_df


def smooth_keypoints(
    kp,
    window=17,
    polyorder=3,
    xcol="pred_x",
    ycol="pred_y",
    x_output_col="pred_x_smooth",
    y_output_col="pred_y_smooth",
):
    # kp: long df with frame, kp_id, u, v
    def _s(g):
        g = g.sort_values("frame").copy()
        w = min(window, len(g) if len(g) % 2 else len(g) - 1)
        if w > polyorder:
            g[x_output_col] = savgol_filter(
                medfilt(g[xcol].values, 5), w, polyorder, mode="interp"
            )
            g[y_output_col] = savgol_filter(
                medfilt(g[ycol].values, 5), w, polyorder, mode="interp"
            )
        return g

    return kp.groupby("class_name", group_keys=False).apply(_s)


# Known table coordinates of each keypoint class (same units as dest_x / dest_y)
DEST_PTS = pd.DataFrame(
    {
        "class_name": [
            "left_1", "left_2", "left_3", "left_baseline",
            "right_1", "right_2", "right_3", "right_baseline",
        ],
        "dest_x": [3, 3, 3, 3, 23, 23, 23, 23],
        "dest_y": [94, 18, 12, 6, 94, 18, 12, 6],
    }
)
MIDLINE_X = DEST_PTS["dest_x"].mean()

DEST_PTS_3D = DEST_PTS.copy()
DEST_PTS_3D['dest_z'] = 0
 
 
def _fit_homography(sub, ransac_thresh):
    """Homography for one frame, or None if the keypoints can't define one reliably."""
    if len(sub) < 4:
        return None
 
    src = sub[["pred_x", "pred_y"]].to_numpy(np.float32)
    dst = sub[["dest_x", "dest_y"]].to_numpy(np.float32)
 
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None or not np.all(np.isfinite(H)):
        return None
 
    # All keypoints on one side lie on a single line, which is degenerate.
    # Require at least 2 inliers on each side (and 4 total).
    inl = mask.ravel().astype(bool)
    dx = dst[inl, 0]
    if inl.sum() < 4 or (dx < MIDLINE_X).sum() < 2 or (dx > MIDLINE_X).sum() < 2:
        return None
    return H
 
 
def apply_homography_to_predictions(
    all_stones_df,
    all_keypoints_df,
    ransac_thresh=0.5,    # in DESTINATION units (same units as dest_x/dest_y); tune
    max_fill_frames=30,   # don't borrow a homography from further away than this
):
    # 1. Keep the HIGHEST-confidence detection per class per frame
    kp = (
        all_keypoints_df.sort_values("confidence", ascending=False)
        .drop_duplicates(["frame", "class_name"], keep="first")
        .merge(DEST_PTS, on="class_name", how="inner")  # drops classes with no known destination
    )
 
    # 2. One homography per frame (or None)
    H_by_frame = {f: _fit_homography(sub, ransac_thresh) for f, sub in kp.groupby("frame")}
    H_by_frame = {int(f): h for f, h in H_by_frame.items() if h is not None}
    if not H_by_frame:
        raise ValueError("No frame had enough valid keypoints to fit a homography.")
 
    stones = all_stones_df.copy()
    stone_frames = stones["frame"].to_numpy(int)
 
    # 3. Put homographies on a contiguous, frame-ordered axis, then fill gaps in frame order
    lo = min(stone_frames.min(), min(H_by_frame))
    hi = max(stone_frames.max(), max(H_by_frame))
    H_flat = (
        pd.DataFrame.from_dict({f: h.ravel() for f, h in H_by_frame.items()}, orient="index")
        .reindex(np.arange(lo, hi + 1))
        .ffill(limit=max_fill_frames)
        .bfill(limit=max_fill_frames)
    )
    H_arr = H_flat.to_numpy(float).reshape(-1, 3, 3)  # NaN where nothing is close enough
 
    # 4. Apply each row's own-frame homography
    H_rows = H_arr[stone_frames - lo]
    coords = np.column_stack(
        [stones["pred_x"].to_numpy(float), stones["pred_y"].to_numpy(float), np.ones(len(stones))]
    )
    out = np.einsum("nij,nj->ni", H_rows, coords)
    w = np.where(np.abs(out[:, 2]) < 1e-9, np.nan, out[:, 2])
 
    stones["x"] = out[:, 0] / w
    stones["y"] = out[:, 1] / w
    stones["h_filled"] = ~stones["frame"].isin(H_by_frame.keys())  # borrowed from another frame
    return stones

def assign_frame_toss_ids(frame_events, stones_in_motion_count, frame_ids, patience=10):
    """patience is measured in FRAME IDs, not rows.
    - Motion fewer than `patience` frames after the last movement continues the toss, and the
      settled/missing rows in between are filled with that toss ID.
    - At `patience` or more frames since the last movement the toss is over. A stone_in_motion
      after that, with no stone_initialized before it, is noise: it gets None.
    - A premature-stone handoff still starts a new toss without stone_initialized."""
    events = list(frame_events)
    counts = list(stones_in_motion_count)
    frames = list(frame_ids)
    assert len(events) == len(counts) == len(frames)
    assert all(b > a for a, b in zip(frames, frames[1:])), "frame_ids must be strictly increasing"

    toss_ids = []
    current, after_premature = 0, False
    toss_active = False             # True from stone_initialized until patience runs out
    last_move_idx = None            # row index of the most recent movement event

    for i, (event, n_moving) in enumerate(zip(events, counts)):
        if event == "stone_initialized":
            current += 1
            after_premature, toss_active = False, True
            toss_ids.append(current)
            last_move_idx = i

        elif event == "premature_stone_on_board_detected":
            after_premature = True
            toss_ids.append(current)
            last_move_idx = i

        elif event == "stone_in_motion":
            prev = current
            starts_new = after_premature and n_moving < 2
            gap_ok = (
                last_move_idx is not None
                and frames[i] - frames[last_move_idx] < patience
            )

            if starts_new:                       # premature handoff: legitimate new toss
                current += 1
                after_premature, toss_active = False, True
                toss_ids.append(current)
                if prev > 0 and gap_ok:          # settled gap before it belongs to the old toss
                    for j in range(last_move_idx + 1, i):
                        toss_ids[j] = prev
                last_move_idx = i

            elif toss_active and gap_ok:         # normal continuation of the current toss
                toss_ids.append(current)
                for j in range(last_move_idx + 1, i):
                    toss_ids[j] = current
                last_move_idx = i

            else:                                # toss already over (or never started): noise
                toss_active = False
                toss_ids.append(None)

        elif event == "all_stones_settled" or event is None:
            toss_ids.append(None)

        else:
            raise ValueError(f"Unknown event: {event!r}")

    return pd.DataFrame({"frame": frames, "toss_id": toss_ids})

# def assign_frame_toss_ids(frame_events, stones_in_motion_count, frame_ids, patience=10):
#     all_toss_ids = []
#     current_toss_id = 1  # initialize the toss ID value
#     after_premature_stone = False
#     already_updated = True
#     frames_since_last_movement = 0
#     after_release = False
    
#     for index, event in enumerate(frame_events):
#         # If the stone is new, it's the start of a toss
#         if event == "stone_initialized":
#             all_toss_ids.append(current_toss_id)
#             already_updated = False
#             frames_since_last_movement = 0
#             after_release = True
#         if event == "all_stones_settled":
#             all_toss_ids.append(None)
#             if frames_since_last_movement >= patience and not already_updated:
#                 print('updating toss ID')
#                 print(frame_ids[index])
#                 current_toss_id = current_toss_id + 1
#                 after_release = False
#                 already_updated = True
#             if frames_since_last_movement < patience:
#                 frames_since_last_movement = frames_since_last_movement + 1
#         if event == "stone_in_motion":
#             if not after_premature_stone and after_release:
#                 all_toss_ids.append(current_toss_id)
#             if after_premature_stone and after_release:
#                 if (
#                     stones_in_motion_count[index] < 2
#                 ):  # Indicates that the premature stone is the only one in motion at this time
#                     print('updating toss ID')
#                     current_toss_id = current_toss_id + 1
#                     after_premature_stone = False
#                     all_toss_ids.append(current_toss_id)
#                 else:
#                     all_toss_ids.append(current_toss_id)
#             if not after_release:
#                 all_toss_ids.append(None)
#             frames_since_last_movement = 0
#         if event == "premature_stone_on_board_detected":
#             all_toss_ids.append(current_toss_id)
#             after_premature_stone = True
#             already_updated = False
#             frames_since_last_movement = 0 # Some other stone was definitively moving at this point
#             after_release = True

    
#     return pd.DataFrame({"frame": frame_ids, "toss_id": all_toss_ids})


def smooth_trajectories(
    g,
    window=17,
    polyorder=3,
    xcol="pred_x",
    ycol="pred_y",
    x_output_col="pred_x_smooth",
    y_output_col="pred_y_smooth",
):
    g = g.sort_values("frame").copy()
    n = len(g)

    # window must be odd and no larger than the trajectory
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= polyorder:  # too short to smooth, leave as is
        g[x_output_col], g[y_output_col] = g[xcol], g[ycol]
        return g

    g[x_output_col] = savgol_filter(g[xcol].values, w, polyorder, mode="interp")
    g[y_output_col] = savgol_filter(g[ycol].values, w, polyorder, mode="interp")
    return g



def add_stone_info(track, minimum_frames_per_stone=10):
    track = track.assign(
        lag_x=lambda x: x.groupby("track_id")["x"].transform("shift"),
        lag_y=lambda x: x.groupby("track_id")["y"].transform("shift"),
        lag_frame=lambda x: x.groupby("track_id")["frame"].transform("shift"),
        n_frames_for_stone=lambda x: x.groupby("track_id")["frame"].transform(
            "nunique"
        ),
    )

    # Remove stones that appear in fewer than X frames (default 10)
    track = track[track["n_frames_for_stone"] > minimum_frames_per_stone]

    # Flag if the stone is inbounds, first do so in the x and y dimensions, then overall
    track["in_bounds_x"] = np.where((track["x"] >= 3.0) & (track["x"] <= 23.0), 1, 0)
    track["in_bounds_y"] = np.where((track["y"] >= 6.0) & (track["y"] <= 176.0), 1, 0)
    track["in_bounds"] = np.where(
        (track["in_bounds_x"] == 0) | (track["in_bounds_y"] == 0), 0, 1
    )

    # Track the stone's distance from the previous frame it appeared in
    track["dist"] = np.sqrt(
        (track["lag_x"] - track["x"]) ** 2 + (track["lag_y"] - track["y"]) ** 2
    )
    # Track the stone's speed from the previous frame it appeared in
    track["s"] = (
        (track["dist"] / 12 / 5280)
        * 30
        * 60
        * 60
        / (track["frame"] - track["lag_frame"])
    )

    # Clean up the data a little more. Make the class names more constant. A stone's color is the most common predicted color while it's in bounds
    track = track.groupby("track_id").apply(
        lambda x: x.assign(
            final_class_name=x["class_name"][(x["in_bounds"] == 1) & (x["confidence"] >= 0.8)].mode().iloc[0]
            if not x["class_name"][(x["in_bounds"] == 1) & (x["confidence"] >= 0.8)].empty
            else None
        )
    )
    
    return track


def clean_tracking_data(
    stone_pred_df,
    keypoint_pred_df,
    video_width,
    video_height,
    minimum_frames_per_stone=10,
    minimum_settled_distance_allowed=1 / 10,
):

    # Map the camera coordinates and 
    K, dist, rms = estimate_intrinsics(keypoint_pred_df, DEST_PTS_3D, (video_width, video_height))   # once per video

    K, med_rms = estimate_focal_length(keypoint_pred_df, DEST_PTS_3D, (video_width, video_height))   # med_rms should be ~1-2 px

    
    print(video_width)
    print(video_height)
    keypoint_pred_df.to_csv('Data/keypoint_pred_df.csv')
    print(K)

    track = apply_3d_adjustment(stone_pred_df, keypoint_pred_df, DEST_PTS_3D, K, dist,
                                stone_height = 1, table_z = 0.0, out_cols = ("x", "y"))

    track = track.reset_index(drop = True)

    # Smooth trajectories (first pass)
    track = smooth_dataframe(track, gaussian_smoother(5), ['track_id'], xcol = 'x', ycol = 'y',
                             out_cols = ['x', 'y'])

    # add things like speed, acceleration, etc
    track = add_stone_info(track, minimum_frames_per_stone)

    # TODO - MOVE THIS DOWN BELOW!!!!
    # Conditions for a stone's score

    # Flag if this stone is settled. Defined by the distance from their previous frame being less than 0.1 inches
    track["stone_settled"] = np.where(
        # (track["dist"] > minimum_settled_distance_allowed) & (track["in_bounds"] == 1) | (track["in_bounds"] == 0),
        (track["dist"] > minimum_settled_distance_allowed), # DO want to include the out of bounds stones for this
        # since the last moving stone going off the board would be 'settled' once it goes off the board and end the 
        # toss early, causing the out of bounds stone to be included incorrectly
        0,
        1,
    )
    # TODO - fix some of these issues with out of bounds stones
    # Flag if this stone is newly initialized. Defined as a stone being in their first frame in the data and being in bounds
    track["stone_initialized"] = np.where(
        (track["dist"].isna()) & (track["in_bounds"] == 1), 1, 0
    )

    # Reset the index
    track = track.reset_index(drop=True)

    # Aggregate the individual stone information by frame
    track_frame_agg = (
        track.groupby("frame")
        .agg(
            # Track how many stones are in the frame, that's defined as the number of rows in the frame group
            stones_in_frame=("track_id", "size"),
            # Track how many stones are settled in the frame. Defined as the sum of the stone_settled column created above
            stones_settled_in_frame=("stone_settled", "sum"),
            # Track how many stones are initialized in the frame. Defined as the sum of the stone_initialized column created above
            # TODO - need to fix what dictates the start of a toss.
            stones_initialized_in_frame=("stone_initialized", "sum"),
        )
        .reset_index()
    )

    # Reset the index
    track = track.reset_index(drop=True)

    # TODO - fix the issues with all_stones_settled event. If a stone loses its track for even a frame, it could cause issues
    # (Need to improve model and botsort parameters to ensure that this rarely, if ever, happens. But need to have corrections in place in case)

    # Conditions for the event for a frame
    conditions = [
        (
            track_frame_agg["stones_initialized_in_frame"] > 0
        ),  # Stone was initialized in the frame
        (
            track_frame_agg["stones_in_frame"]
            == track_frame_agg["stones_settled_in_frame"]
        ),  # All stones were settled in the frame
        (
            track_frame_agg["stones_settled_in_frame"]
            < track_frame_agg["stones_in_frame"]
        ),  # At least one stone was in motion in this frame
        (
            (track_frame_agg["stones_initialized_in_frame"] > 0)
            & (
                track_frame_agg["stones_settled_in_frame"]
                < track_frame_agg["stones_in_frame"]
            )
        ),  # A stone was initialized and there are stones in motion on the board
    ]

    # Choices for the event for a frame
    choices = [
        "stone_initialized",
        "all_stones_settled",
        "stone_in_motion",
        "premature_stone_on_board_detected",
    ]
    # Select the event that meeets the criteria from the above conditions
    track_frame_agg["event"] = np.select(conditions, choices, default=None)
    track_frame_agg["stones_in_motion"] = (
        track_frame_agg["stones_in_frame"] - track_frame_agg["stones_settled_in_frame"]
    )

    track_frame_agg.to_csv("Data/test_frame_agg.csv")

    # merge the track event info with the original tracking data
    track = pd.merge(track, track_frame_agg, on="frame", how="left")

    # Flag the row where a premature stone is initialized
    track["premature_toss_flag"] = np.where(
        (track["stone_initialized"] == 1)
        & (track["event"] == "premature_stone_on_board_detected")
        & (track['in_bounds'] == 1),
        1,
        0,
    )
    # TODO - how better to handle this part? Need the redundancy in place to handle these

    # New process
    # Toss starts when:
    # 1. all stones are settled and stone initialized
    # 2. a stone is initialized and not all stones are settled, the new toss will begin when the other stones do settle
    #
    # Toss ends when:
    # 1. All stones settle
    # 2. All stones except for a stone with invalid release are settled.

    # Sort the data
    track.sort_values(
        by=["frame", "stone_initialized", "track_id"],
        ascending=[True, False, True],
        inplace=True,
    )

    # Find the lag event
    track["lag_event"] = track["event"].shift(1)

    track["stones_newly_settled_flag"] = np.where(
        (track["event"] == "all_stones_settled")
        & (track["lag_event"] != "all_stones_settled"),
        1,
        0,
    )

    # Flag if a new toss began on this frame
    track["start_new_toss_flag"] = np.where(
        (track["event"] == "stone_initialized")
        & (track["lag_event"] != "stone_initialized"),
        1,
        0,
    )

    # Flag if this is the start or an end of a toss
    track["round_event_flag"] = np.where(
        (track["start_new_toss_flag"] == 1) | (track["stones_newly_settled_flag"] == 1),
        1,
        0,
    )

    #
    conditions = [
        (track["stones_newly_settled_flag"] == 1),
        (track["start_new_toss_flag"] == 1),
    ]

    choices = ["settled", "start"]

    track["start_or_settled_last"] = np.select(conditions, choices, default=None)

    track["start_or_settled_last"] = track["start_or_settled_last"].ffill()

    track.sort_values(
        by=["frame", "stone_initialized", "track_id"],
        ascending=[True, False, True],
        inplace=True,
    )

    track["cumsum"] = track["start_new_toss_flag"].cumsum()

    # TODO - determine the toss IDs

    # track['toss_id'] = np.where(track['start_or_settled_last'] == 'start', track['cumsum'], None)

    new_toss_ids = assign_frame_toss_ids(
        track_frame_agg["event"].tolist(),
        track_frame_agg["stones_in_motion"].tolist(),
        track_frame_agg["frame"].tolist(),
    )

    track = pd.merge(track, new_toss_ids, how="left", on="frame")

    # track = track.groupby(["track_id", "toss_id"], group_keys=False).apply(
    #     smooth_trajectory, process_noise=0.01, measurement_noise=5.0
    # )

    # Recalculate these after smoothing
    track = track.assign(
        lag_x=lambda x: x.groupby(["track_id", "toss_id"])["x"].transform("shift"),
        lag_y=lambda x: x.groupby(["track_id", "toss_id"])["y"].transform("shift"),
        lag_frame=lambda x: x.groupby(["track_id", "toss_id"])["frame"].transform(
            "shift"
        ),
        n_frames_for_stone=lambda x: x.groupby(["track_id", "toss_id"])[
            "frame"
        ].transform("nunique"),
    )
    track["dist"] = np.sqrt(
        (track["lag_x"] - track["x"]) ** 2 + (track["lag_y"] - track["y"]) ** 2
    )
    track["s"] = (
        (track["dist"] / 12 / 5280)
        * 30
        * 60
        * 60
        / (track["frame"] - track["lag_frame"])
    )

    conditions = [
        (track["y"].between(6, 12)) & (track["in_bounds"] == 1),  # 3 points
        (track["y"].between(12, 18)) & (track["in_bounds"] == 1),  # 2 points
        (track["y"].between(18, 94)) & (track["in_bounds"] == 1),  # 1 point
        # TODO - may need to adjust this last condition in case of estimation error
        (track["y"] > 94) & (track["in_bounds"] == 1),  # 0 points
        track["in_bounds"] == 0,  # 0 points
    ]

    # Choices for a stone's score
    choices = [3, 2, 1, 0, 0]

    # Select the score for a given stone based on the criteria above
    track["stone_score"] = np.select(conditions, choices, default=0)
    
    track = track.groupby('track_id').apply(
        lambda x: x.assign(
            # If a stone is in bounds and above the halfcourt line, then it's the stone that was tossed
            track_id_last_frame_before_crossing_half_court = x["frame"][(x["in_bounds"] == 1) & (x["y"] >= 96)].max()
        )
    )

    track = track.groupby('toss_id').apply(
        lambda x: x.assign(
            # If a stone is in bounds and above the halfcourt line, then it's the stone that was tossed
            toss_id_last_frame_before_crossing_half_court = x['track_id_last_frame_before_crossing_half_court'].max()
        )
    )

    # TODO - better way of determining this: go off of each stone's last frame where y > 96. Most recent stone that was 
    #        above the half-court line is the tossed stone. This will fix the index out of bounds issue
    # Note that the criteria is 96 and not 94. This is to provide a little leeway in case the models say that a stone directly on the halfcourt line is actually above it
    track["is_tossed_stone"] = np.where(track["track_id_last_frame_before_crossing_half_court"] == track["toss_id_last_frame_before_crossing_half_court"], 1, 0)

    track = track.reset_index(drop=True)

    track.to_csv('Data/early_tracking_data.csv')

    return track


def cross2d(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def track_wembo_status_helper(track, toss_id, minimum_settled_distance_allowed=1 / 10):
    track_sub = track[track["toss_id"] == toss_id]
    track_sub = track_sub[track_sub["in_bounds"] == 1]

    # TODO - add logic for SGA stones. An SGA splitting two stones would only be a wembo if they split their opponent's stones, not their own
    # TODO - add guardrails for Josh nonchalant gimmick where he puts stone on the table before preceding stone settles
    # TODO - build out the 'no contact with another stone' criterion more accurately (this will be a v1.1 update)
    # If some other stone (aside from the tossed stone) moved at some point during the frame, it cannot be a wembo
    if (
        len(
            track_sub[
                (track_sub["dist"] >= minimum_settled_distance_allowed)
                & (track_sub["is_tossed_stone"] == 0)
            ]
        )
        > 0
    ):
        print("ineligible")
        return pd.DataFrame({"toss_id": toss_id, "wembo_split": 0}, index=[0])

    # If no stone other than the tossed stone moved, find the order of them
    track_sub_first_row = track_sub.groupby("track_id").head(1)

    if len(track_sub_first_row[track_sub_first_row["is_tossed_stone"] == 1]) == 0:
        print("toss error")
        return pd.DataFrame({"toss_id": toss_id, "wembo_split": 0}, index=[0])

    toss_class = track_sub_first_row[track_sub_first_row["is_tossed_stone"] == 1][
        "final_class_name"
    ].iloc[0]

    # If there are fewer than 2 stones for the opposing stone color on the board, skip the rest of this
    if (
        len(track_sub_first_row[track_sub_first_row["final_class_name"] != toss_class])
        < 2
    ):
        print("impossible")
        return pd.DataFrame({"toss_id": toss_id, "wembo_split": 0}, index=[0])

    x_sorted = track_sub_first_row.sort_values("x")
    y_sorted = track_sub_first_row.sort_values("y")

    # First handle x direction
    x_track_x_values = x_sorted["x"].tolist()
    x_track_y_values = x_sorted["y"].tolist()
    x_track_class = x_sorted["final_class_name"].tolist()

    # Then handle y direction
    y_track_x_values = y_sorted["x"].tolist()
    y_track_y_values = y_sorted["y"].tolist()
    y_track_class = y_sorted["final_class_name"].tolist()

    eligible_wembo_pairs = []

    for i in range(1, len(x_track_class)):
        if x_track_class[i] != toss_class and x_track_class[i - 1] != toss_class:
            pair = (
                (x_track_x_values[i], x_track_y_values[i]),
                (x_track_x_values[i - 1], x_track_y_values[i - 1]),
            )

            eligible_wembo_pairs.append(pair)

    for i in range(1, len(y_track_class)):
        if y_track_class[i] != toss_class and y_track_class[i - 1] != toss_class:
            pair = (
                (y_track_x_values[i], y_track_y_values[i]),
                (y_track_x_values[i - 1], y_track_y_values[i - 1]),
            )

            eligible_wembo_pairs.append(pair)

    if len(eligible_wembo_pairs) == 0:
        print("no adjacent pairs")
        return pd.DataFrame({"toss_id": toss_id, "wembo_split": 0}, index=[0])

    tossed_stone_subset = track_sub[
        (track_sub["is_tossed_stone"] == 1) & (track_sub["lag_x"].notna())
    ]

    print(tossed_stone_subset)
    for wembo_pair in eligible_wembo_pairs:
        print(wembo_pair)
        for index, row in tossed_stone_subset.iterrows():
            # print(row['lag_x'])
            a = (row["lag_x"], row["lag_y"])
            b = (row["x"], row["y"])

            comp_point_1 = wembo_pair[0]
            comp_point_2 = wembo_pair[1]

            d1 = cross2d(comp_point_1, comp_point_2, a)
            d2 = cross2d(comp_point_1, comp_point_2, b)
            d3 = cross2d(a, b, comp_point_1)
            d4 = cross2d(a, b, comp_point_2)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
                (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
            ):
                print(row["frame"])
                return pd.DataFrame({"toss_id": toss_id, "wembo_split": 1}, index=[0])

    return pd.DataFrame({"toss_id": toss_id, "wembo_split": 0}, index=[0])


def track_wembo_status(track, minimum_settled_distance_allowed=1 / 10):
    # TODO
    # Should return a dataframe with two columns: toss ID and wembo, which is a binary flag that denotes if a toss was a wembo or not

    all_toss_ids = track["toss_id"].unique()

    all_wembo_tracks = pd.DataFrame()
    for wembo_toss_id in all_toss_ids:
        toss_wembo_track = track_wembo_status_helper(
            track=track,
            toss_id=wembo_toss_id,
            minimum_settled_distance_allowed=minimum_settled_distance_allowed,
        )
        all_wembo_tracks = pd.concat([all_wembo_tracks, toss_wembo_track])

    return all_wembo_tracks


def build_all_data_formats(
    stone_pred_df, keypoint_pred_df, video_width, video_height, minimum_frames_per_stone=2, minimum_settled_distance_allowed=1 / 10
):
    # TODO

    # Clean the tracking data
    track_cleaned = clean_tracking_data(stone_pred_df, keypoint_pred_df, video_width, video_height)

    # Remove the data that's not part of a toss
    track_tosses = track_cleaned[track_cleaned["toss_id"] != None]

    print(track_tosses["toss_id"].unique())

    track_wembos = track_wembo_status(track_tosses)

    # This is causing issues again after building in the kalman filter and radial distortion correction.
    test_track = track_tosses.groupby("toss_id").agg(
        has_any_tossed=("is_tossed_stone", "max")
    )

    print(test_track)
    print(track_tosses[["toss_id", "track_id"]].drop_duplicates())

    # Find the max frame value in each toss ID and merge that column into the tracking data
    final_frame_df = track_tosses.groupby("toss_id").agg(
        max_frame_in_toss=("frame", "max"),
        class_of_tossed_stone=(
            "final_class_name",
            lambda x: x[track_tosses.loc[x.index, "is_tossed_stone"] == 1].iloc[0],
        ),
    )

    track_tosses_final_frame = pd.merge(
        track_tosses, final_frame_df, how="left", on="toss_id"
    )

    # Filter down to the final frame, and only include stones that are on the board
    track_tosses_final_frame = track_tosses_final_frame[
        track_tosses_final_frame["frame"]
        == track_tosses_final_frame["max_frame_in_toss"]
    ]
    track_tosses_final_frame = track_tosses_final_frame[
        track_tosses_final_frame["in_bounds"] == 1
    ]

    track_tosses_final_frame = track_tosses_final_frame.assign(
        lag_stone_score=lambda x: x.groupby("track_id")["stone_score"].transform(
            "shift"
        ),
    )

    track_tosses_final_frame["favor"] = np.where(
        (
            track_tosses_final_frame["stone_score"]
            > track_tosses_final_frame["lag_stone_score"]
        )
        & (
            track_tosses_final_frame["final_class_name"]
            != track_tosses_final_frame["class_of_tossed_stone"]
        ),
        1,
        0,
    )

    track_tosses_final_frame["assist"] = np.where(
        (
            track_tosses_final_frame["stone_score"]
            > track_tosses_final_frame["lag_stone_score"]
        )
        & (
            track_tosses_final_frame["final_class_name"]
            == track_tosses_final_frame["class_of_tossed_stone"]
        ),
        1,
        0,
    )

    toss_results = track_tosses_final_frame.groupby("toss_id").agg(
        class_of_tossed_stone=("class_of_tossed_stone", "first"),
        points_from_toss=(
            "stone_score",
            lambda x: (
                x[track_tosses_final_frame.loc[x.index, "is_tossed_stone"] == 1].iloc[0]
                if x[
                    track_tosses_final_frame.loc[x.index, "is_tossed_stone"] == 1
                ].any()
                else 0
            ),
        ),
        black_stones_on_board=(
            "final_class_name",
            lambda x: (x == "black_stone").sum(),
        ),
        gray_stones_on_board=("final_class_name", lambda x: (x == "gray_stone").sum()),
        favor=("favor", "max"),
        assist=("assist", "max"),
    )

    track_tosses_final_frame.drop(columns="class_of_tossed_stone", inplace=True)

    toss_results = pd.merge(toss_results, track_wembos, how="left", on="toss_id")
    toss_results["wembo"] = np.where(
        (toss_results["wembo_split"] == 1) & (toss_results["points_from_toss"] == 0),
        1,
        0,
    )

    toss_results.drop(columns="wembo_split", inplace=True)
    # Calculate the scores after each toss
    track_tosses_final_frame_scores = track_tosses_final_frame.groupby(
        ["toss_id", "final_class_name"]
    ).agg(window_score=("stone_score", "sum"))

    # TODO - create table for individual toss results
    return (
        track_tosses,
        track_tosses_final_frame,
        track_tosses_final_frame_scores,
        toss_results,
    )


def build_data_from_game_folder(video_path):
    yaml_file_path = video_path + "/game_config.yaml"

    with open(yaml_file_path, "r") as file:
        game_config = yaml.safe_load(file)

    black_stone_player = game_config["black_stone"]
    gray_stone_player = game_config["gray_stone"]

    videos = []
    video_time_stamps = []
    video_widths = []
    video_heights = []
    for file_name in os.listdir(video_path):
        if file_name.lower().endswith((".mp4", ".mov", ".avi")):
            full_video_path = video_path + "/" + file_name
            video_stats = os.stat(full_video_path)
            video_time_stamp = datetime.fromtimestamp(video_stats.st_mtime)
            cap = cv2.VideoCapture(full_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(video_stats.st_mtime)
            videos.append(full_video_path)
            video_time_stamps.append(video_time_stamp)
            video_widths.append(width)
            video_heights.append(height)

    print(videos)
    print(video_time_stamps)
    sorted_videos = [x for _, x in sorted(zip(video_time_stamps, videos))]
    print(sorted_videos)

    video_time_stamps.sort()
    game_start_str = video_time_stamps[0].strftime("%Y-%m-%d %H:%M:%S")

    print(game_start_str)

    window_number = 1
    black_stone_overall_score_last_window = 0
    gray_stone_overall_score_last_window = 0
    full_game_track = pd.DataFrame()
    full_game_final_frames = pd.DataFrame()
    full_game_round_scores = pd.DataFrame()
    for index, video in enumerate(sorted_videos):
        print(video + " " + str(window_number))
        stone_pred_df, keypoint_pred_df = predict_new_video(video_path=video, window=window_number)

        (
            cleaned_pred_track_window,
            pred_final_frame,
            pred_round_scores,
            pred_toss_scores,
        ) = build_all_data_formats(stone_pred_df, keypoint_pred_df, video_width = video_widths[index], video_height = video_heights[index])

        # Don't drop this index since that is the toss ID field
        pred_round_scores = pred_round_scores.reset_index()

        print(pred_round_scores)

        pred_round_scores = pred_round_scores.pivot(
            index="toss_id", columns="final_class_name", values="window_score"
        ).fillna(0)

        pred_round_scores = pred_round_scores.rename(
            columns={
                "black_stone": "black_stone_window_score",
                "gray_stone": "gray_stone_window_score",
            }
        )

        pred_round_scores = pd.merge(
            pred_round_scores, pred_toss_scores, how="left", on="toss_id"
        )

        # TODO - reindex the frame ID, right now when it's visualized, it cuts back and forth like a Christopher Nolan movie
        # Add 3 columns: one for the video's timestamp, the window, and the time when it was last updated
        # Add columns for the tracking data
        cleaned_pred_track_window["game_timestamp"] = game_start_str
        cleaned_pred_track_window["window"] = window_number
        cleaned_pred_track_window["last_updated"] = pd.Timestamp.now()

        # Add columns for the final frame data
        pred_final_frame["game_timestamp"] = game_start_str
        pred_final_frame["window"] = window_number
        pred_final_frame["last_updated"] = pd.Timestamp.now()

        # Add columns for the round by round data
        pred_round_scores["game_timestamp"] = game_start_str
        pred_round_scores["window"] = window_number
        pred_round_scores["last_updated"] = pd.Timestamp.now()

        pred_round_scores["black_stone_overall_score"] = (
            pred_round_scores["black_stone_window_score"]
            + black_stone_overall_score_last_window
        )
        pred_round_scores["gray_stone_overall_score"] = (
            pred_round_scores["gray_stone_window_score"]
            + gray_stone_overall_score_last_window
        )

        black_stone_overall_score_last_window = pred_round_scores[
            "black_stone_overall_score"
        ].iloc[-1]
        gray_stone_overall_score_last_window = pred_round_scores[
            "gray_stone_overall_score"
        ].iloc[-1]

        # Append the data from the individual window
        full_game_track = pd.concat([full_game_track, cleaned_pred_track_window])
        full_game_final_frames = pd.concat([full_game_final_frames, pred_final_frame])
        full_game_round_scores = pd.concat([full_game_round_scores, pred_round_scores])

        window_number = window_number + 1

    full_game_round_scores.reset_index(inplace=True, drop=True)
    full_game_final_frames.reset_index(inplace=True, drop=True)
    full_game_round_scores.reset_index(inplace=True, drop=True)
    return full_game_track, full_game_final_frames, full_game_round_scores


# track, final_frame, pbp = build_data_from_game_folder("Film/Andy_Mike")

# print(track)
# print(final_frame)
# print(pbp)

# track.to_csv("Data/test_window_track.csv")
# final_frame.to_csv("Data/test_window_final_frames.csv")
# pbp.to_csv("Data/test_window_pbp_scores.csv")


def predict_all_games(path_to_all_videos):
    # TODO - make a data frame that will have all of the data appended
    for folder in os.listdir(path_to_all_videos):
        if os.path.isdir(path_to_all_videos + "/" + folder):
            print(folder)
            # TODO


predict_all_games("Film")


# TODO - build function to push all data to a database

# Edge Case Game: 2-1 black after first window | 6-2 gray final

# TODO - figure out what's causing all of the jitter.
# possible causes:
# - keypoint smoothing is overkill
# - current smoothing adjustment for stones needs finetuning
