import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import json
import os
import base64
from datetime import datetime

# CONFIG 
MODEL_PATH = "yolo_pallet_project/igps_detector/weights/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODEL
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# PAGE CONFIG & STYLING
st.set_page_config(page_title="iGPS Pallet Detector", layout="centered")

st.markdown("""
    <style>
        .main {background-color: #f4f4f8;}
        h1 {
            text-align: center;
            color: #2c3e50;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .subheader {
            text-align: center;
            font-size: 1rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
        }
        .stDownloadButton>button {
            background-color: #1f4e79;
            color: white;
        }
        .confidence-slider {
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# TITLE 
st.markdown("<h1>📦 iGPS Pallet Detector</h1>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Upload an image to detect <strong>iGPS pallets</strong> in marketplace listings</div>", unsafe_allow_html=True)

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# CONFIDENCE SLIDER
st.markdown("<div class='confidence-slider'>", unsafe_allow_html=True)
confidence = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Filter out detections below this confidence level")
st.markdown("</div>", unsafe_allow_html=True)

# INFERENCE
if uploaded_file:
    with st.spinner("🔍 Running YOLOv8 inference..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(image, conf=confidence, device=DEVICE)
        result = results[0]
        annotated_image = result.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Extract detection details
        detections = []
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            detections.append({
                "class": model.names[int(cls)],
                "confidence": round(conf, 2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        # OUTPUT DISPLAY
        if detections:
            st.image(annotated_image_rgb, caption="🖼️ Detected iGPS Pallets", use_container_width=True)

            st.markdown("### 🧾 Detection Details")
            st.dataframe(detections, use_container_width=True)

            json_data = json.dumps({
                "filename": uploaded_file.name,
                "timestamp": datetime.now().isoformat(),
                "detections": detections
            }, indent=4)

            json_filename = f"results_{os.path.splitext(uploaded_file.name)[0]}.json"
            b64 = base64.b64encode(json_data.encode()).decode()

            with st.expander("📄 View JSON Result"):
                st.code(json_data, language="json")

            st.download_button(
                label="📥 Download Detection JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("🚫 No iGPS pallets detected in this image.")
else:
    st.info("📤 Upload a marketplace image to begin.")
