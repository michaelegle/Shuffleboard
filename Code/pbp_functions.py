# Functions for building the data

from ultralytics import YOLO
from ultralytics.trackers import bot_sort
from custom_botsort import DistanceAwareBOTSORT
from custom_kalman_filter_params import CollisionAwareKalmanFilter
from types import MethodType
import time
import numpy as np
import cv2
import pandas as pd
import yaml
import os
from datetime import datetime

SOURCE = "Film/test_clip.MOV"
TRACKER = "Code/custom_botsort_params.yaml"
# TODO - fix the keypoint tracking parameters
KEYPOINT_TRACKER = "Code/keypoint_tracking_params.yaml"

def predict_new_video(video_path,
                      window,
                      model_dir = "Models/stone_detection/model_saves/weights/best.pt"):

    stone_model = YOLO(stone_model_dir)
    keypoint_model = YOLO(keypoint_model_dir)
    # Step 1: stream=True gives us a generator — grab just the first frame
    #         so Ultralytics builds the predictor AND registers trackers
    generator = stone_model.track(
        source = video_path,
        tracker = TRACKER,
        persist = True,
        device = "mps",
        stream = True, 
        save = False
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
    start = time.perf_counter()
    stone_results = stone_model.track(
        source = video_path,
        tracker = TRACKER,
        iou = 0.5,
        save = False,
        save_json = False,
        persist = True,
        device = "mps",
        stream = True
    )

    keypoint_results = keypoint_model.predict(
        source = video_path,
        save = True,
        conf = 0.25,
        save_json = True,
        device = "mps",
        stream = True
    )


    all_predictions_df = pd.DataFrame()
    for frame_idx, result in enumerate(results):
        frame_predictions = pd.DataFrame()
        for box in result.boxes:
            prediction = {
                "frame": frame_idx,
                "class_id":   int(box.cls),
                "class_name": stone_model.names[int(box.cls)],
                "confidence": float(box.conf),
                "track_id":   int(box.id) if box.id is not None else None,
                "pred_x": float(box.xywh[0][0]),
                "pred_y": float(box.xywh[0][1])
            }
            # TODO - find issue from frames 400-600 being missed for some reason
            prediction_df = pd.DataFrame(prediction, index = [0])
            frame_predictions = pd.concat([frame_predictions, prediction_df])
        all_predictions_df = pd.concat([all_predictions_df, frame_predictions])

    all_keypoints_df = pd.DataFrame()
    for frame_idx, result in enumerate(keypoint_results):
        frame_predictions = pd.DataFrame()
        for box in result.boxes:
            prediction = {
                "frame": frame_idx,
                "class_id":   int(box.cls),
                "class_name": keypoint_model.names[int(box.cls)],
                "confidence": float(box.conf),
                "pred_x": float(box.xywh[0][0]),
                "pred_y": float(box.xywh[0][1])
            }
            prediction_df = pd.DataFrame(prediction, index = [0])
            frame_predictions = pd.concat([frame_predictions, prediction_df])
        all_keypoints_df = pd.concat([all_keypoints_df, frame_predictions])
    
    
    points = np.vstack([all_predictions_df['pred_x'], 
                        all_predictions_df['pred_y'], 
                        np.ones(len(all_predictions_df['pred_x']))])
    
    print(points)


    # TODO - this is temporary and just used for testing. In the future, use the keypoint detection model to find
    #        the keypoints for the homography matrix
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

    # Destination (ground truth) points for where each point should be mapped to
    pts_dest = np.array([[3, 94], [3, 18], [3, 12], [3, 6],
                         [23, 94], [23, 18], [23, 12], [23, 6]])

    # Calculate the homography matrix
    h = cv2.findHomography(pts_source, pts_dest, cv2.RANSAC)
    h = h[0]

    # Transform and standardize all of the points
    transformed_points = h @ points
    x_new = transformed_points[0] / transformed_points[2]
    y_new = transformed_points[1] / transformed_points[2]

    # Create a new column for the x and y columns after the homography matrix has been applied
    all_predictions_df['x'] = x_new
    all_predictions_df['y'] = y_new

    #all_predictions_df.to_csv("Data/new_predictions.csv")

    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")

    return all_predictions_df


def clean_tracking_data(track, 
                        minimum_frames_per_stone = 10,
                        minimum_settled_distance_allowed = 1 / 16):

    track = track.assign(
        lag_x = lambda x: x.groupby('track_id')['x'].transform('shift'),
        lag_y = lambda x: x.groupby('track_id')['y'].transform('shift'),
        lag_frame = lambda x: x.groupby('track_id')['frame'].transform('shift'),
        n_frames_for_stone = lambda x: x.groupby('track_id')['frame'].transform('nunique')
    )
    
    # TODO - address the camera perspective making stones look farther from the camera than they really are

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

    # TODO - fix the issues with all_stones_settled event. If a stone loses its track for even a frame, it could cause issues
    # (Need to improve model and botsort parameters to ensure that this rarely, if ever, happens. But need to have corrections in place in case)

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

    # Find the lag event
    track['lag_event'] = track['event'].shift(1)
    # TODO - fix the stones newly settled flag
    track['stones_newly_settled_flag'] = np.where((track['event'] == 'all_stones_settled') & (track['lag_event'] != 'all_stones_settled'), 1, 0)

    # Flag if a new toss began on this frame
    track['start_new_toss_flag'] = np.where((track['event'] == 'stone_initialized') & (track['lag_event'] != 'stone_initialized'), 1, 0)
    
    # Flag if this is the start or an end of a toss
    track['round_event_flag'] = np.where((track['start_new_toss_flag'] == 1) | (track['stones_newly_settled_flag'] == 1), 1, 0)

    # 
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

def cross2d(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def track_wembo_status_helper(track, toss_id, minimum_settled_distance_allowed = 1 / 16):
    track_sub = track[track['toss_id'] == toss_id]

    # If some other stone (aside from the tossed stone) moved at some point during the frame, it cannot be a wembo
    if len(track_sub[(track_sub['stone_settled'] == 0) & (track_sub['is_tossed_stone'] == 0)]) > 0:
        return pd.DataFrame({'toss_id' : toss_id,
                             'wembo_split' : 0},
                             index = [0])
    
    
    # If no stone other than the tossed stone moved, find the order of 
    track_sub_first_row = track_sub.groupby('track_id').head(1)

    toss_class = track_sub_first_row[track_sub_first_row['is_tossed_stone'] == 1]['final_class_name'].iloc[0]

    # If there are fewer than 2 stones for the opposing stone color on the board, skip the rest of this
    if(len(track_sub_first_row[track_sub_first_row['final_class_name'] != toss_class]) < 2):
        return pd.DataFrame({'toss_id' : toss_id,
                             'wembo_split' : 0},
                             index = [0])

    x_sorted = track_sub_first_row.sort_values('x')
    y_sorted = track_sub_first_row.sort_values('y')

    # First handle x direction
    x_track_x_values = x_sorted['x'].tolist()
    x_track_y_values = x_sorted['y'].tolist()
    x_track_class = x_sorted['final_class_name'].tolist()

    # Then handle y direction
    y_track_x_values = y_sorted['x'].tolist()
    y_track_y_values = y_sorted['y'].tolist()
    y_track_class = y_sorted['final_class_name'].tolist()

    eligible_wembo_pairs = []

    for i in range(1, len(x_track_class)):
        if (x_track_class[i] != toss_class and x_track_class[i - 1] != toss_class):
            pair = ((x_track_x_values[i], x_track_y_values[i]),
                    (x_track_x_values[i - 1], x_track_y_values[i - 1]))
            
            eligible_wembo_pairs.append(pair)

    for i in range(1, len(y_track_class)):
        if (y_track_class[i] != toss_class and y_track_class[i - 1] != toss_class):
            pair = ((y_track_x_values[i], y_track_y_values[i]),
                    (y_track_x_values[i - 1], y_track_y_values[i - 1]))
            
            eligible_wembo_pairs.append(pair)

    if(len(eligible_wembo_pairs) == 0):
        return pd.DataFrame({'toss_id' : toss_id,
                             'wembo_split' : 0},
                             index = [0])
    
    tossed_stone_subset = track_sub_first_row[(track_sub_first_row['is_tossed_stone'] == 1) & (track_sub_first_row['lag_x'].notna())]

    for wembo_pair in eligible_wembo_pairs:
        # TODO
        for row in tossed_stone_subset.iterrows():
            a = (row['lag_x'], row['lag_y'])
            b = (row['x'], row['y'])

            comp_point_1 = wembo_pair[0]
            comp_point_2 = wembo_pair[1]

            d1 = cross2d(comp_point_1, comp_point_2, a)
            d2 = cross2d(comp_point_1, comp_point_2, b)
            d3 = cross2d(a, b, comp_point_1)
            d4 = cross2d(a, b, comp_point_2)
            if (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
                ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))):
                return pd.DataFrame({'toss_id' : toss_id,
                                     'wembo_split' : 1},
                                     index = [0])

    return pd.DataFrame({'toss_id' : toss_id,
                         'wembo_split' : 0},
                         index = [0])





def track_wembo_status(track, minimum_settled_distance_allowed = 1 / 16):
    # TODO
    # Should return a dataframe with two columns: toss ID and wembo, which is a binary flag that denotes if a toss was a wembo or not

    all_toss_ids = track['toss_id'].unique()

    all_wembo_tracks = pd.DataFrame()
    for wembo_toss_id in all_toss_ids:
        toss_wembo_track = track_wembo_status_helper(track = track, toss_id = wembo_toss_id, minimum_settled_distance_allowed = minimum_settled_distance_allowed)
        all_wembo_tracks = pd.concat([all_wembo_tracks, toss_wembo_track])

    return all_wembo_tracks

def build_all_data_formats(track, 
                           minimum_frames_per_stone = 2,
                           minimum_settled_distance_allowed = 1 / 16):
    # TODO 

    # Clean the tracking data
    track_cleaned = clean_tracking_data(track)

    # Remove the data that's not part of a toss
    track_tosses = track_cleaned[track_cleaned['toss_id'] != None]

    track_wembos = track_wembo_status(track_tosses)

    # Find the max frame value in each toss ID and merge that column into the tracking data
    final_frame_df = track_tosses.groupby('toss_id').agg(
        max_frame_in_toss = ('frame', 'max'),
        class_of_tossed_stone = ('final_class_name', 
                                 lambda x: x[track_tosses.loc[x.index, 'is_tossed_stone'] == 1].iloc[0]))
    
    track_tosses_final_frame = pd.merge(track_tosses, final_frame_df, how = "left", on = 'toss_id')
    
    # Filter down to the final frame, and only include stones that are on the board
    track_tosses_final_frame = track_tosses_final_frame[track_tosses_final_frame['frame'] == track_tosses_final_frame['max_frame_in_toss']]
    track_tosses_final_frame = track_tosses_final_frame[track_tosses_final_frame['in_bounds'] == 1]

    track_tosses_final_frame = track_tosses_final_frame.assign(
        lag_stone_score = lambda x: x.groupby('track_id')['stone_score'].transform('shift'),
    )

    track_tosses_final_frame['favor'] = np.where((track_tosses_final_frame['stone_score'] > track_tosses_final_frame['lag_stone_score']) & 
                                                 (track_tosses_final_frame['final_class_name'] != track_tosses_final_frame['class_of_tossed_stone']),
                                                 1, 0)
    
    track_tosses_final_frame['assist'] = np.where((track_tosses_final_frame['stone_score'] > track_tosses_final_frame['lag_stone_score']) & 
                                                  (track_tosses_final_frame['final_class_name'] == track_tosses_final_frame['class_of_tossed_stone']),
                                                  1, 0)

    toss_results = track_tosses_final_frame.groupby('toss_id').agg(
        class_of_tossed_stone = ('class_of_tossed_stone', 'first'),
        points_from_toss = ('stone_score', lambda x:
                            x[track_tosses_final_frame.loc[x.index, 'is_tossed_stone'] == 1].iloc[0]
                            if x[track_tosses_final_frame.loc[x.index, 'is_tossed_stone'] == 1].any()
                            else 0),
        black_stones_on_board = ('final_class_name', lambda x: (x == 'black_stone').sum()),
        gray_stones_on_board = ('final_class_name', lambda x: (x == 'gray_stone').sum()),
        favor = ('favor', 'max'),
        assist = ('assist', 'max')
    )

    track_tosses_final_frame.drop(columns = 'class_of_tossed_stone', inplace = True)

    toss_results = pd.merge(toss_results, track_wembos, how = 'left', on = 'toss_id')
    toss_results['wembo'] = np.where((toss_results['wembo_split'] == 1) & (toss_results['points_from_toss'] == 0), 1, 0)

    toss_results.drop(columns = 'wembo_split', inplace = True)
    # Calculate the scores after each toss
    track_tosses_final_frame_scores = track_tosses_final_frame.groupby(['toss_id', 'final_class_name']).agg(window_score = ('stone_score', 'sum'))

    # TODO - create table for individual toss results
    return track_tosses, track_tosses_final_frame, track_tosses_final_frame_scores, toss_results



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
    black_stone_overall_score_last_window = 0
    gray_stone_overall_score_last_window = 0
    full_game_track = pd.DataFrame()
    full_game_final_frames = pd.DataFrame()
    full_game_round_scores = pd.DataFrame()
    for video in sorted_videos:

        print(video + " " + str(window_number))
        pred_track_window = predict_new_video(video_path = video,
                                              window = window_number)


        cleaned_pred_track_window, pred_final_frame, pred_round_scores, pred_toss_scores = build_all_data_formats(track = pred_track_window)


        # Don't drop this index since that is the toss ID field
        pred_round_scores = pred_round_scores.reset_index()

        print(pred_round_scores)

        pred_round_scores = pred_round_scores.pivot(index = 'toss_id', columns = 'final_class_name', values = 'window_score').fillna(0)

        pred_round_scores = pred_round_scores.rename(columns = {'black_stone': 'black_stone_window_score',
                                                                'gray_stone': 'gray_stone_window_score'})

        
        pred_round_scores = pd.merge(pred_round_scores, pred_toss_scores, how = "left", on = "toss_id")

        # Add 3 columns: one for the video's timestamp, the window, and the time when it was last updated
        # Add columns for the tracking data
        cleaned_pred_track_window['game_timestamp'] = game_start_str
        cleaned_pred_track_window['window'] = window_number
        cleaned_pred_track_window['last_updated'] = pd.Timestamp.now()

        # Add columns for the final frame data
        pred_final_frame['game_timestamp'] = game_start_str
        pred_final_frame['window'] = window_number
        pred_final_frame['last_updated'] = pd.Timestamp.now()

        # Add columns for the round by round data
        pred_round_scores['game_timestamp'] = game_start_str
        pred_round_scores['window'] = window_number
        pred_round_scores['last_updated'] = pd.Timestamp.now()

        
        pred_round_scores['black_stone_overall_score'] = pred_round_scores['black_stone_window_score'] + black_stone_overall_score_last_window
        pred_round_scores['gray_stone_overall_score'] = pred_round_scores['gray_stone_window_score'] + gray_stone_overall_score_last_window


        black_stone_overall_score_last_window = pred_round_scores['black_stone_overall_score'].iloc[-1]
        gray_stone_overall_score_last_window = pred_round_scores['gray_stone_overall_score'].iloc[-1]


        # Append the data from the individual window
        full_game_track = pd.concat([full_game_track, cleaned_pred_track_window])
        full_game_final_frames = pd.concat([full_game_final_frames, pred_final_frame])
        full_game_round_scores = pd.concat([full_game_round_scores, pred_round_scores])


        
        window_number = window_number + 1

    full_game_round_scores.reset_index(inplace = True, drop = True)
    full_game_final_frames.reset_index(inplace = True, drop = True)
    full_game_round_scores.reset_index(inplace = True, drop = True)
    return full_game_track, full_game_final_frames, full_game_round_scores

track, final_frame, pbp = build_data_from_game_folder("Film/Andy_Kyle")

print(track)
print(final_frame)
print(pbp)

track.to_csv('Data/test_window_track.csv')
final_frame.to_csv('Data/test_window_final_frames.csv')
pbp.to_csv('Data/test_window_pbp_scores.csv')


def predict_all_games(path_to_all_videos):
    # TODO - make a data frame that will have all of the data appended
    for folder in os.listdir(path_to_all_videos):
        if os.path.isdir(path_to_all_videos + '/' + folder):
            print(folder)
            # TODO

predict_all_games('Film')