# Wembo test file

import pbp_functions as pbp_func
import pandas as pd
import matplotlib.pyplot as plt

event_frames = pd.read_csv('Data/test_frame_agg.csv')

print(event_frames)

result = pbp_func.assign_frame_toss_ids(event_frames['event'].tolist(),
                                        event_frames['stones_in_motion'].tolist(),
                                        event_frames['frame'].tolist())

print(result)

result.to_csv('Data/output_test.csv')

