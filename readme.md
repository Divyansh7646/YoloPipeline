# iGPS Pallet Detection Pipeline

This project detects iGPS pallets using a combination of:

- YOLOv8 – for object detection (detecting pallets)
- EasyOCR – for text recognition (verifying "iGPS" text inside the bounding boxes)
- Designed as a clean demo pipeline to showcase modern computer vision techniques

---

## Features

- Trains a YOLOv8 model for iGPS pallet detection
- Runs inference on new images using the trained model
- Applies OCR to detected pallet regions to confirm the presence of "iGPS" text
- Categorizes matches into:
  - **strong** → exact "iGPS"
  - **partial** → similar text (like "igp", "gps")
  - **ambiguous** → possible misreads ("icps", "1gps")
- Saves annotated images and prints detection summaries

---

## Project Structure

```
project-root/
├── yolov8_pallet/           # Dataset + scripts
│   ├── data.yaml            # YOLO dataset config file
│   ├── train.py        # Training script (YOLOv8)
│   ├── infer.py        # Inference script (YOLOv8)
│   └── infer2.py # EasyOCR script for logo detection
├── requirements.txt         # Required dependencies
├── README.md                # Project description
└── yolo_pallet_project/
    └── igps_detector/
        └── weights/
            └── best.pt      # Trained YOLOv8 model weights


---

## Setup Instructions (for VS Code)

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Unzip and verify dataset

Extract your Roboflow-exported dataset (ZIP file) into the `yolov8_pallet/` directory. Ensure the `data.yaml` file is correctly configured with paths to train, val, and test sets.

## Training the YOLOv8 Model

Run the following command to start training:

```bash
python yolo_train.py
```

The training script:
- Uses `yolov8m.pt` (medium model) for better accuracy
- Saves weights in `yolo_pallet_project/igps_detector/weights/`

## Running Inference with YOLOv8

To test the trained model on a new image:

```bash
python yolo_infer.py
```

When prompted, enter the path to your input image. The script will display the image with bounding boxes and save an annotated version as `inference_result.jpg`.

## Running OCR for iGPS Text Detection

To verify if the detected pallet contains "iGPS":

```bash
python ocr_igps_detector.py
```

When prompted, enter the path to the image. The script will:
- Detect text within the image
- Classify text as strong, partial, or ambiguous
- Save the annotated output in `inference_output/`

## Requirements

All dependencies are listed in `requirements.txt`. Key packages include:

- ultralytics (YOLOv8)
- torch
- opencv-python
- matplotlib
- easyocr
- numpy

Install them using:

```bash
pip install -r requirements.txt
```

## Outputs

- YOLO training results and logs are saved in `yolo_pallet_project/igps_detector/`
- Inference results are saved as `inference_result.jpg`
- OCR detection outputs are saved in `inference_output/`

## Notes

- Ensure CUDA is available for GPU acceleration (training will be slow on CPU)
- Adjust parameters like `EPOCHS`, `BATCH_SIZE`, and `IMG_SIZE` in `yolo_train.py` as needed