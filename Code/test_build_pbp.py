
import pandas as pd
import numpy as np
import math

track_raw = pd.read_csv("Data/predictions.csv")

def clean_tracking_data(track):

    track = track.assign(
        lag_x = lambda x: x.groupby('track_id')['x'].transform('shift'),
        lag_y = lambda x: x.groupby('track_id')['y'].transform('shift')
    )

    track['in_bounds_x'] = np.where((track['x'] >= 3.) & (track['x'] <= 23.), 1, 0)
    track['in_bounds_y'] = np.where((track['y'] >= 6.) & (track['y'] <= 176.), 1, 0)
    track['in_bounds'] = np.where((track['in_bounds_x'] == 0) | (track['in_bounds_y'] == 0), 0, 1)

    track['dist'] = np.sqrt((track['lag_x'] - track['x'])**2 + (track['lag_y'] - track['y'])**2)
    track['s'] = (track['dist'] / 12 / 5280) * 30 * 60 * 60


    track = track.groupby('track_id').apply(lambda x: x.assign(
        final_class_name = x['class_name'][x['in_bounds'] == 1].mode()[0]
    ))

    conditions = [
        (track['y'].between(6, 12)) & (track['in_bounds'] == 1),
        (track['y'].between(12, 18)) & (track['in_bounds'] == 1),
        (track['y'].between(18, 94)) & (track['in_bounds'] == 1),
        (track['y'] > 94) & (track['in_bounds'] == 1),
        track['in_bounds'] == 0
    ]

    choices = [3, 2, 1, 0, 0]

    track['stone_score'] = np.select(conditions, choices, default=0)

    track['stone_settled'] = np.where(track['dist'] < 0.08, 1, 0)
    track['stone_initialized'] = np.where(track['dist'].isna(), 1, 0)

    track = track.reset_index(drop = True)

    track_frame_agg = track.groupby('frame').agg(
        stones_in_frame = ('frame', 'size'),
        stones_settled_in_frame = ('stone_settled', 'sum'),
        stones_initialized_in_frame = ('stone_initialized', 'sum')
    )

    track = track.reset_index(drop = True)

    conditions = [
        (track_frame_agg['stones_initialized_in_frame'] > 0),
        (track_frame_agg['stones_in_frame'] == track_frame_agg['stones_settled_in_frame']),
        (track_frame_agg['stones_initialized_in_frame'] == 0) & (track_frame_agg['stones_settled_in_frame'] < track_frame_agg['stones_in_frame'])
    ]

    choices = ['stone_initialized', 'all_stones_settled', 'stone_in_motion']
    track_frame_agg['event'] = np.select(conditions, choices, default = None)
    
    track = pd.merge(track, track_frame_agg, on = 'frame', how = 'left')

    track.sort_values(by = ['frame', 'stone_initialized', 'track_id'], ascending = [True, False, True], inplace = True)

    track['lag_event'] = track['event'].shift(1)
    track['stones_newly_settled_flag'] = np.where((track['event'] == 'all_stones_settled') & (track['lag_event'] != 'all_stones_settled'), 1, 0)

    track['start_new_toss_flag'] = np.where((track['event'] == 'stone_initialized') & (track['lag_event'] != 'stone_initialized'), 1, 0)
    track['round_event_flag'] = np.where((track['start_new_toss_flag'] == 1) | (track['start_new_toss_flag'] == 1), 1, 0)

    track.sort_values(by = ['frame', 'stone_initialized', 'track_id'], ascending = [True, False, True], inplace = True)

    track['cumsum'] = track['round_event_flag'].cumsum()
    track['during_toss'] = np.where(track['cumsum'] % 2 == 1, 1, 0)

    track['toss_id'] = np.where(track['during_toss'] == 1, track['cumsum'], None)

    track = track.reset_index(drop = True)

    return track

track_cleaned = clean_tracking_data(track_raw)

print(track_cleaned)

print(track_cleaned['stone_settled'])

track_cleaned_agg = track_cleaned.groupby(['frame', 'final_class_name']).agg(total_score = ('stone_score', 'sum'))

print(track_cleaned_agg)

track_cleaned.to_csv("Data/tracking_data_cleaned.csv")
