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

model_dir = "Code/runs/detect/train10/weights/best.pt"
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


points = np.vstack([all_predictions_df['pred_x'], 
                    all_predictions_df['pred_y'], 
                    np.ones(len(all_predictions_df['pred_x']))])

print(points)


# Left 1 point line: 244, 975 -> 0, 88
# Left 2 point line: 109, 482 -> 0, 12
# Left 3 point line: 82, 373 -> 0, 6
# Left baseline: 46, 248 -> 0, 0

# Right 1 point line: 445, 975 -> 20, 88
# Right 2 point line: 573, 488 -> 20, 12
# Right 3 point line: 599, 382 -> 20, 6
# Right baseline: 634, 253 -> 20, 0

pts_source = np.array([[244, 1280 - 975], [109, 1280 - 482], [82, 1280 - 373], [46, 1280 - 248],
                       [445, 1280 - 975], [573, 1280 - 488], [599, 1280 - 382], [634, 1280 - 253]])

pts_dest = np.array([[3, 94], [3, 18], [3, 12], [3, 6],
                     [23, 94], [23, 18], [23, 12], [23, 6]])

h = cv2.findHomography(pts_source, pts_dest, cv2.RANSAC)

h = h[0]

H = np.array([[   0.027449,   -0.007561,     0.64209],
              [ -0.0005194,    0.033204,     -8.1587],
              [ -5.636e-06, -0.00074346,           1]])

transformed_points = h @ points

print(h)

x_new = transformed_points[0] / transformed_points[2]
y_new = transformed_points[1] / transformed_points[2]

all_predictions_df['x'] = x_new
all_predictions_df['y'] = y_new

all_predictions_df.to_csv("Data/predictions.csv")

end = time.perf_counter()
print(f"Execution time: {end - start:.6f} seconds")


