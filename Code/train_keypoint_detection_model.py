from ultralytics import YOLO
import torch
import multiprocessing

# Load a pretrained keypoint model
model = YOLO("yolov8n.pt")

print(torch.cuda.is_available())

def train_model():
    # Train
    model.train(
        data="Models/keypoint_detection/data.yaml",
        project = "C:/Users/Michael Egle/Documents/Shuffleboard",
        name = "Models/keypoint_detection/model_saves",
        imgsz=640,
        epochs=100,
        batch=16,
        lr0=0.001,
        degrees = 10,
        scale=0.5,
        shear=2.0,
        mosaic=1.0,       # fine to keep, doesn't mirror content
        mixup=0.1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        device=0
    )

if __name__ == '__main__':
    p = multiprocessing.Process(target = train_model)
    p.start()
    p.join()

