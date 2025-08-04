import os
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# CONFIG
MODEL_PATH = "yolo_pallet_project/igps_detector/weights/best.pt"

# INPUT
image_path = input(" Enter path to image for inference: ").strip()

if not os.path.exists(image_path):
    raise FileNotFoundError(f" Image not found: {image_path}")

# LOAD MODEL
print(" Loading model...")
model = YOLO(MODEL_PATH)

# RUN INFERENCE
print(" Running inference...")
results = model(image_path, conf=0.5)

# DISPLAY RESULT
result = results[0]

# Draw results on image
annotated_frame = result.plot()  # Returns image with bounding boxes

# Convert BGR to RGB for matplotlib
annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

# Show using matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(annotated_frame_rgb)
plt.axis("off")
plt.title(" Inference Result")
plt.show()

# SAVE IMAGE
save_path = "inference_result.jpg"
cv2.imwrite(save_path, annotated_frame)
print(f" Result saved to: {save_path}")
