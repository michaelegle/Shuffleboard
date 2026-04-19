import time
from ultralytics import YOLO
import json

model_dir = "Code/runs/detect/train10/weights/best.pt"

model = YOLO(model_dir)

start = time.perf_counter()
results = model.predict(source = "Film/IMG_9788.MOV", tracker="bytetrack.yaml", save = True, save_json = True)
end = time.perf_counter()


print(f"Execution time: {end - start:.6f} seconds")

all_predictions = []

for frame_idx, result in enumerate(results):
    frame_data = {
        "frame": frame_idx,
        "predictions": []
    }
    
    box_ids = result.boxes.id
    box_idx = 0

    for box in result.boxes:
        prediction = {
            "class_id": int(box.cls),
            "instance_id": box_ids[box_idx],
            "class_name": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": {
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3])
            }
        }
        frame_data["predictions"].append(prediction)
        box_idx = box_idx + 1
    
    all_predictions.append(frame_data)

with open("Data/predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=2)