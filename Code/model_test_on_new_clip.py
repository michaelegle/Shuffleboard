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

model_dir = "Models/keypoint_detection/model_saves/weights/best.pt"
model = YOLO(model_dir)

SOURCE = "Film/test_clip.MOV"

# Step 4: full inference with patches active
start = time.perf_counter()
results = model.predict(
    source=SOURCE,
    save=True,
    conf = 0.25,
    save_json=True,
    device="mps",
    stream=True
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
            "pred_x": float(box.xywh[0][0]),
            "pred_y": float(box.xywh[0][1])
        }
        prediction_df = pd.DataFrame(prediction, index = [0])
        frame_predictions = pd.concat([frame_predictions, prediction_df])
    all_predictions_df = pd.concat([all_predictions_df, frame_predictions])

all_predictions_df = all_predictions_df.sort_values('confidence').drop_duplicates(['frame', 'class_name'])

print(all_predictions_df)

dest_pts = pd.DataFrame({
    'class_name': ['left_1', 'left_2', 'left_3', 'left_baseline',
                 'right_1', 'right_2', 'right_3', 'right_baseline'],
    'dest_x': [3, 3, 3, 3,
               23, 23, 23, 23],
    'dest_y': [94, 18, 12, 6,
               94, 18, 12, 6]
})

all_predictions_df = pd.merge(all_predictions_df, dest_pts, how = 'left', on = 'class_name')


print(all_predictions_df)
