from ultralytics import YOLO
import torch
import os

# CONFIG 
DATA_YAML = "yolov8_pallet/data.yaml"
MODEL_ARCH = "yolov8m.pt"  
IMG_SIZE = 640             
EPOCHS = 50
BATCH_SIZE = -1
PROJECT = "yolo_pallet_project"
NAME = "igps_detector"
PATIENCE = 10
WORKERS = 4

# GPU CHECK 
print(" CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(" Using GPU:", torch.cuda.get_device_name(0))
else:
    print(" CUDA not available. Training on CPU...")

# TRAINING
def main():
    model = YOLO(MODEL_ARCH)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device="cuda",
        project=PROJECT,
        name=NAME,
        amp=True,
        patience=PATIENCE,
        workers=WORKERS,
        exist_ok=True,
        verbose=True,
        cos_lr=True,
        seed=42,
        val=True
    )

    print(" Training complete. Model saved to:", os.path.join(PROJECT, NAME, "weights"))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    main()
