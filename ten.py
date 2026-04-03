
import os
os.add_dll_directory(r"D:\Rushikesh\TensorRT-10.16.0.72\lib")
os.add_dll_directory(r"D:\Rushikesh\TensorRT-10.16.0.72\bin")

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image



# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Container Damage Detection", layout="wide")
st.title("🚢 Container Damage Detection (YOLO)")

# Sidebar
st.sidebar.header("Model Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35)
iou_thres = st.sidebar.slider("IoU Threshold", 0.3, 0.9, 0.6)

# Load model
@st.cache_resource
def load_model():
    # 🔥 ONLY CHANGE: .pt → .engine
    return YOLO(r"D:\Rushikesh\project\conkalmarYOLO26\used\KalmarV9.engine")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a container image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Image Inference
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run inference (IMPORTANT FIXES HERE)
    results = model.predict(
        source=image_np,
        imgsz=640,          # 🔥 MUST match TensorRT engine (was 1024)
        conf=confidence,
        iou=iou_thres,
        max_det=50,
        save=False,
        verbose=False,
        device=0            # 🔥 ensure GPU
    )

    # Draw results
    annotated_img = results[0].plot()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detected Regions")
        st.image(annotated_img, use_column_width=True)

    # Detection info
    st.subheader("Detection Details")
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf_score = float(boxes.conf[i])
            class_name = model.names[cls_id]

            st.write(
                f"**Detection {i+1}:** `{class_name}` | Confidence: `{conf_score:.2f}`"
            )
    else:
        st.warning("No damage detected. Try lowering confidence.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("YOLO-based Industrial Damage Detection")