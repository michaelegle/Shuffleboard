from projection_3d import _best_keypoints, estimate_focal_length
import pandas as pd

keypoints_df = pd.read_csv("Data/keypoint_pred_df.csv")

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

width = 720
height = 1280

dest_df = DEST_PTS.copy()
dest_df['dest_z'] = 0

best = _best_keypoints(keypoints_df, dest_df)
print(len(best), best.groupby("frame").size().describe())   # expect ~7-8 per frame
K, med = estimate_focal_length(keypoints_df, dest_df, (width, height))
print(K[0, 0], med) 