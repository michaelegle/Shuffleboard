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
TRACKER = "Code/custom_botsort_params.yaml"

# Step 1: stream=True gives us a generator — grab just the first frame
#         so Ultralytics builds the predictor AND registers trackers
generator = model.track(
    source=SOURCE,
    tracker=TRACKER,
    persist=True,
    device="mps",
    stream=True,        # critical — don't process whole video yet
    save=False
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
    source=SOURCE,
    tracker=TRACKER,
    save=True,
    save_json=True,
    persist=True,
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
            "track_id":   int(box.id) if box.id is not None else None,
            "pred_x": float(box.xywh[0][0]),
            "pred_y": float(box.xywh[0][1])
        }
        prediction_df = pd.DataFrame(prediction, index = [0])
        frame_predictions = pd.concat([frame_predictions, prediction_df])
    all_predictions_df = pd.concat([all_predictions_df, frame_predictions])

print(all_predictions_df)