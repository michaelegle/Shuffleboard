import time
from ultralytics import YOLO

model_dir = "C:/Users/Michael Egle/Documents/Shuffleboard/Code/runs/detect/train10/weights/best.pt"

model = YOLO(model_dir)

start = time.perf_counter()
results = model.predict(source = "../Film/IMG_9777.MOV")
end = time.perf_counter()

print(f"Execution time: {end - start:.6f} seconds")