
import pandas as pd
import numpy as np
import math

track_raw = pd.read_csv("Data/predictions.csv")

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
        final_class_name = x['class_name'][x['in_bounds'] == 1].mode()[0]
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

track_cleaned = clean_tracking_data(track_raw)

print(track_cleaned)

print(track_cleaned['stone_settled'])

track_frame_cleaned_agg = track_cleaned.groupby(['frame', 'final_class_name']).agg(total_score = ('stone_score', 'sum'))

print(track_frame_cleaned_agg)

track_cleaned.to_csv("Data/tracking_data_cleaned.csv")

track_tosses = track_cleaned[track_cleaned['toss_id'] != None]

final_frame_df = track_tosses.groupby('toss_id').agg(max_frame_in_toss = ('frame', 'max'))

track_tosses_final_frame = pd.merge(track_tosses, final_frame_df, how = "left", on = 'toss_id')


track_tosses_final_frame = track_tosses_final_frame[track_tosses_final_frame['frame'] == track_tosses_final_frame['max_frame_in_toss']]

track_tosses_final_frame = track_tosses_final_frame[track_tosses_final_frame['in_bounds'] == 1]


track_tosses_final_frame.to_csv("Data/tracking_data_cleaned_final_frame.csv")

print(track_tosses_final_frame)


track_tosses_final_frame_scores = track_tosses_final_frame.groupby(['toss_id', 'final_class_name']).agg(total_score = ('stone_score', 'sum'))

print(track_tosses_final_frame_scores)

# 4 Levels of data, in increasing granularity:
# 1. Game Result Data (Player names, results, dates, etc)
# 2. Toss by toss data. Score at the end of each toss, number of tosses taken and remaining for either player
# 3. Board state data. Stone locations at the end of each toss
# 4. Tracking data. Each stone's location at each moment during the throw

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




