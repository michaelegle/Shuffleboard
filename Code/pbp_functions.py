# Functions for building the data

from ultralytics import YOLO
from ultralytics.trackers import bot_sort
from custom_botsort import DistanceAwareBOTSORT
from custom_kalman_filter_params import CollisionAwareKalmanFilter
from types import MethodType
import time
import json
import numpy as np
import cv2
import pandas as pd
import yaml
import os
from datetime import datetime

SOURCE = "Film/test_clip.MOV"
TRACKER = "Code/custom_botsort_params.yaml"

def predict_new_video(video_path,
                      window,
                      model_dir = "Code/runs/detect/train10/weights/best.pt"):

    model = YOLO(model_dir)
    # Step 1: stream=True gives us a generator — grab just the first frame
    #         so Ultralytics builds the predictor AND registers trackers
    generator = model.track(
        source = video_path,
        tracker = TRACKER,
        persist = True,
        device = "mps",
        stream = True, 
        save = False
    )

    first_result = next(generator)  # just one frame to init predictor.trackers

    # Step 2: now trackers exists — patch the live instance
    tracker = model.predictor.trackers[0]
    tracker.__class__ = DistanceAwareBOTSORT
    tracker.kalman_filter.__class__ = CollisionAwareKalmanFilter
    tracker.reset()  # wipe state from the dummy frame
    
    # Step 3: close the generator, run fresh on the full video
    generator.close()
    
    # Step 4: full inference with patches active
    start = time.perf_counter()
    results = model.track(
        source = video_path,
        tracker = TRACKER,
        save = True,
        save_json = False,
        persist = True,
        device = "mps",
        stream = True
    )
    
    all_predictions_df = pd.DataFrame()
    for frame_idx, result in enumerate(results):
        frame_data = {"frame": frame_idx, "predictions": []}
        frame_predictions = pd.DataFrame()
        for box in result.boxes:
            prediction = {
                "frame": frame_idx,
                "class_id":   int(box.cls),
                "class_name": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "track_id":   int(box.id) if box.id is not None else None,
                "pred_x": float(box.xywh[0][0]),
                "pred_y": float(box.xywh[0][1])
            }
            prediction_df = pd.DataFrame(prediction, index = [0])
            frame_predictions = pd.concat([frame_predictions, prediction_df])
        all_predictions_df = pd.concat([all_predictions_df, frame_predictions])
    
    
    points = np.vstack([all_predictions_df['pred_x'], 
                        all_predictions_df['pred_y'], 
                        np.ones(len(all_predictions_df['pred_x']))])
    
    print(points)

    if "Andy_Mike" in video_path:
        if window == 1:
            pts_source = np.array([[312, 415], [176, 801], [151, 881], [118, 976],
                                   [490, 424], [581, 824], [601, 907], [623, 1010]])
        if window == 2:
            pts_source = np.array([[282, 428], [155, 826], [129, 908], [100, 1009],
                                   [466, 439], [566, 839], [589, 920], [613, 1026]])

    if "Andy_Kyle" in video_path:
        if window == 1:
            pts_source = np.array([[255, 449], [130, 906], [101, 1006], [70, 1131],
                                   [438, 455], [582, 916], [613, 1013], [648, 1143]])
        if window == 2:
            pts_source = np.array([[272, 436], [154, 875], [127, 971], [97, 1091],
                                   [461, 446], [594, 890], [625, 987], [658, 1106]])
    
    if "Kyle_Mike" in video_path:
        if window == 1:
            pts_source = np.array([[256, 427], [131, 844], [104, 927], [73, 1037],
                                   [442, 437], [552, 855], [578, 941], [604, 1050]])
        if window == 2:
            pts_source = np.array([[236, 439], [111, 920], [83, 1025], [51, 1161],
                                   [429, 446], [577, 921], [614, 1025], [649, 1159]])

    pts_dest = np.array([[3, 94], [3, 18], [3, 12], [3, 6],
                         [23, 94], [23, 18], [23, 12], [23, 6]])

    h = cv2.findHomography(pts_source, pts_dest, cv2.RANSAC)

    h = h[0]

    transformed_points = h @ points

    x_new = transformed_points[0] / transformed_points[2]
    y_new = transformed_points[1] / transformed_points[2]

    all_predictions_df['x'] = x_new
    all_predictions_df['y'] = y_new

    #all_predictions_df.to_csv("Data/new_predictions.csv")

    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")

    return all_predictions_df


def clean_tracking_data(track, 
                        minimum_frames_per_stone = 2,
                        minimum_settled_distance_allowed = 1 / 16):

    track = track.assign(
        lag_x = lambda x: x.groupby('track_id')['x'].transform('shift'),
        lag_y = lambda x: x.groupby('track_id')['y'].transform('shift'),
        lag_frame = lambda x: x.groupby('track_id')['frame'].transform('shift'),
        n_frames_for_stone = lambda x: x.groupby('track_id')['frame'].transform('nunique')
    )

    # Remove stones that appear in fewer than X frames (default 2)
    track = track[track['n_frames_for_stone'] > minimum_frames_per_stone]

    # Flag if the stone is inbounds, first do so in the x and y dimensions, then overall
    track['in_bounds_x'] = np.where((track['x'] >= 3.) & (track['x'] <= 23.), 1, 0)
    track['in_bounds_y'] = np.where((track['y'] >= 6.) & (track['y'] <= 176.), 1, 0)
    track['in_bounds'] = np.where((track['in_bounds_x'] == 0) | (track['in_bounds_y'] == 0), 0, 1)

    # Track the stone's distance from the previous frame it appeared in
    track['dist'] = np.sqrt((track['lag_x'] - track['x'])**2 + (track['lag_y'] - track['y'])**2)
    # Track the stone's speed from the previous frame it appeared in
    track['s'] = (track['dist'] / 12 / 5280) * 30 * 60 * 60 / (track['frame'] - track['lag_frame'])

    # Clean up the data a little more. Make the class names more constant. A stone's color is the most common predicted color while it's in bounds
    track = track.groupby('track_id').apply(lambda x: x.assign(
        final_class_name = x['class_name'][x['in_bounds'] == 1].mode().iloc[0]
        if not x['class_name'][x['in_bounds'] == 1].empty
        else None
    ))

    # Conditions for a stone's score
    conditions = [
        (track['y'].between(6, 12)) & (track['in_bounds'] == 1), # 3 points
        (track['y'].between(12, 18)) & (track['in_bounds'] == 1), # 2 points
        (track['y'].between(18, 94)) & (track['in_bounds'] == 1), # 1 point
        # TODO - may need to adjust this last condition in case of estimation error
        (track['y'] > 94) & (track['in_bounds'] == 1), # 0 points
        track['in_bounds'] == 0 # 0 points
    ]

    # Choices for a stone's score
    choices = [3, 2, 1, 0, 0]

    # Select the score for a given stone based on the criteria above
    track['stone_score'] = np.select(conditions, choices, default=0)

    # Flag if this stone is settled. Defined by the distance from their previous frame being less than 0.1 inches
    track['stone_settled'] = np.where(track['dist'] < minimum_settled_distance_allowed, 1, 0)
    # Flag if this stone is newly initialized. Defined as a stone being in their first frame in the data and being in bounds
    track['stone_initialized'] = np.where((track['dist'].isna()) & (track['in_bounds'] == 1), 1, 0)

    # Reset the index
    track = track.reset_index(drop = True)

    # Aggregate the individual stone information by frame
    track_frame_agg = track.groupby('frame').agg(
        # Track how many stones are in the frame, that's defined as the number of rows in the frame group
        stones_in_frame = ('frame', 'size'),
        # Track how many stones are settled in the frame. Defined as the sum of the stone_settled column created above
        stones_settled_in_frame = ('stone_settled', 'sum'),
        # Track how many stones are initialized in the frame. Defined as the sum of the stone_initialized column created above
        stones_initialized_in_frame = ('stone_initialized', 'sum')
    )

    # Reset the index
    track = track.reset_index(drop = True)

    # Conditions for the event for a frame
    conditions = [
        (track_frame_agg['stones_initialized_in_frame'] > 0), # Stone was initialized in the frame
        (track_frame_agg['stones_in_frame'] == track_frame_agg['stones_settled_in_frame']), # All stones were settled in the frame
        (track_frame_agg['stones_initialized_in_frame'] == 0) & (track_frame_agg['stones_settled_in_frame'] < track_frame_agg['stones_in_frame']) # At least one stone was in motion in this frame
    ]

    # Choices for the event for a frame
    choices = ['stone_initialized', 'all_stones_settled', 'stone_in_motion']
    # Select the event that meeets the criteria from the above conditions
    track_frame_agg['event'] = np.select(conditions, choices, default = None)
    
    # merge the track event info with the original tracking data
    track = pd.merge(track, track_frame_agg, on = 'frame', how = 'left')

    # Sort the data
    track.sort_values(by = ['frame', 'stone_initialized', 'track_id'], ascending = [True, False, True], inplace = True)

    # 
    track['lag_event'] = track['event'].shift(1)
    track['stones_newly_settled_flag'] = np.where((track['event'] == 'all_stones_settled') & (track['lag_event'] != 'all_stones_settled'), 1, 0)

    track['start_new_toss_flag'] = np.where((track['event'] == 'stone_initialized') & (track['lag_event'] != 'stone_initialized'), 1, 0)
    track['round_event_flag'] = np.where((track['start_new_toss_flag'] == 1) | (track['stones_newly_settled_flag'] == 1), 1, 0)

    conditions = [
        (track['stones_newly_settled_flag'] == 1),
        (track['start_new_toss_flag'] == 1)
    ]

    choices = ['settled', 'start']

    track['start_or_settled_last'] = np.select(conditions, choices, default = None)

    track['start_or_settled_last'] = track['start_or_settled_last'].ffill()

    track.sort_values(by = ['frame', 'stone_initialized', 'track_id'], ascending = [True, False, True], inplace = True)

    track['cumsum'] = track['start_new_toss_flag'].cumsum()

    track['toss_id'] = np.where(track['start_or_settled_last'] == 'start', track['cumsum'], None)

    
    track = track.groupby(['track_id', 'toss_id']).apply(lambda x: x.assign(
        # If a stone is in bounds and above the halfcourt line, then it's the stone that was tossed
        track_id_max_y_on_board = x['y'][x['in_bounds'] == 1].max()
    ))

    # Note that the criteria is 96 and not 94. This is to provide a little leeway in case the models say that a stone directly on the halfcourt line is actually above it
    track['is_tossed_stone'] = np.where(track['track_id_max_y_on_board'] > 96, 1, 0)
    
    track = track.reset_index(drop = True)

    return track


def build_all_data_formats(track, 
                           minimum_frames_per_stone = 2,
                           minimum_settled_distance_allowed = 1 / 16):
    # TODO 
    track_cleaned = clean_tracking_data(track)

    track_tosses = track_cleaned[track_cleaned['toss_id'] != None]
    final_frame_df = track_tosses.groupby('toss_id').agg(max_frame_in_toss = ('frame', 'max'))
    track_tosses_final_frame = pd.merge(track_tosses, final_frame_df, how = "left", on = 'toss_id')
    track_tosses_final_frame = track_tosses_final_frame[track_tosses_final_frame['frame'] == track_tosses_final_frame['max_frame_in_toss']]
    track_tosses_final_frame = track_tosses_final_frame[track_tosses_final_frame['in_bounds'] == 1]
    track_tosses_final_frame_scores = track_tosses_final_frame.groupby(['toss_id', 'final_class_name']).agg(total_score = ('stone_score', 'sum'))


    # TODO - figure out the first/second window parts
    return track_tosses, track_tosses_final_frame, track_tosses_final_frame_scores



def build_data_from_game_folder(video_path):
    yaml_file_path = video_path + "/game_config.yaml"

    with open(yaml_file_path, 'r') as file:
        game_config = yaml.safe_load(file)

    black_stone_player = game_config['black_stone']
    gray_stone_player = game_config['gray_stone']

    videos = []
    video_time_stamps = []
    for file_name in os.listdir(video_path):
        if file_name.lower().endswith((".mp4", ".mov", ".avi")):
            full_video_path = video_path + '/' + file_name
            video_stats = os.stat(full_video_path)
            video_time_stamp = datetime.fromtimestamp(video_stats.st_mtime)
            print(video_stats.st_mtime)
            videos.append(full_video_path)
            video_time_stamps.append(video_time_stamp)

    print(videos)
    print(video_time_stamps)
    sorted_videos = [x for _, x in sorted(zip(video_time_stamps, videos))]
    print(sorted_videos)

    video_time_stamps.sort()
    game_start_str = video_time_stamps[0].strftime("%Y-%m-%d %H:%M:%S")

    print(game_start_str)

    window_number = 1
    for video in sorted_videos:

        print(video + " " + str(window_number))
        pred_track_window = predict_new_video(video_path = video,
                                              window = window_number)

        #tracking_data, final_frame_data, frame_scores_data = build_all_data_formats(video)

        cleaned_pred_track_window = clean_tracking_data(pred_track_window)

        cleaned_pred_track_window['game_timestamp'] = game_start_str
        cleaned_pred_track_window['window'] = window_number

        cleaned_pred_track_window['last_updated'] = pd.Timestamp()

        cleaned_pred_track_window.to_csv('Data/test_window.csv')

        window_number = window_number + 1

build_data_from_game_folder("Film/Andy_Kyle")