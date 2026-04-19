from ultralytics import YOLO
import torch
import multiprocessing

# Load a pretrained keypoint model
model = YOLO("yolov8n.pt")

print(torch.cuda.is_available())

def train_model():
    # Train
    model.train(
        data="data.yaml",
        imgsz=640,
        epochs=100,
        batch=16,
        lr0=0.001,
        hsv_h=0.0,  # disable heavy augmentation
        hsv_s=0.0,
        hsv_v=0.0,
        device=0
    )

if __name__ == '__main__':
    p = multiprocessing.Process(target = train_model)
    p.start()
    p.join()

