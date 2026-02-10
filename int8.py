# int8_api.py
"""
Container OCR API — YOLO -> Qwen3-VL (OpenVINO INT8 weight-only)
- Processor/tokenizer loads from HF (or local HF-style dir)
- Model weights (OpenVINO IR) loaded from INT8 export folder
- Keeps YOLO detection and file saving structure
"""

import os
# make sure frameworks do not try to use CUDA accidentally
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Thread / BLAS limits (Windows-safe defaults)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

import time
import json
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import threading

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ultralytics YOLO
from ultralytics import YOLO

# Optimum + OpenVINO (CPU path)
from transformers import AutoProcessor
from optimum.intel import OVModelForVisualCausalLM

# -----------------------------
# CONFIG - adjust paths here
# -----------------------------
# OpenVINO IR export directory (INT8 weight-only export)
QWEN_OV_PATH = r"E:\ocr\flocr\qwen3-vl-2b-int8-weightonly-ov"

# Hugging Face model ID or local HF-style folder (for processor/tokenizer)
# Use the HF model id if you didn't export processor files to the local ov folder.
QWEN_HF_PATH = "Qwen/Qwen3-VL-2B-Instruct"

# YOLO weights (your trained models)
YOLO_PRIMARY = r"D:\Rushikesh\project\ContainerModel_22_01_26_ubuntu1.pt"
YOLO_SECONDARY = r"D:\Rushikesh\project\ContainerModel_12_01_26_3.pt"

# concurrency
QWEN_MAX_CONCURRENCY = 2

# FastAPI run host/port (if running directly)
HOST = "0.0.0.0"
PORT = 8082

# -----------------------------
# Ensure result folders exist
# -----------------------------
for directory in [
    "container_results/received_frames",
    "container_results/yolo_detections",
    "container_results/qwen_images",
    "container_results/success"
]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Pydantic model
# -----------------------------
class PickupEvent(BaseModel):
    kalmar_id: str
    action: str
    timestamp: str
    images: Optional[List[str]] = None
    image_base64: Optional[str] = None

# -----------------------------
# Utility: base64 -> cv2 BGR
# -----------------------------
def base64_to_cv2(b64_string: str) -> np.ndarray:
    try:
        if b64_string.startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]
        image_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None - invalid image data")
        return frame
    except Exception as e:
        print(f"[ERROR] base64_to_cv2 failed: {e}")
        raise

# -----------------------------
# Save helpers
# -----------------------------
def today_folder(base: str) -> str:
    folder_path = os.path.join(base, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_received_frame(frame: np.ndarray, kalmar_id: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(today_folder("container_results/received_frames"), f"{kalmar_id}_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[SAVE] 📷 Received -> {path}")
    return path

def save_yolo_detection(frame: np.ndarray, kalmar_id: str, detected: bool):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    status = "detected" if detected else "no_detection"
    path = os.path.join(today_folder("container_results/yolo_detections"), f"{kalmar_id}_{timestamp}_{status}.jpg")
    cv2.imwrite(path, frame)
    print(f"[SAVE] 🎯 YOLO -> {path}")
    return path

def save_qwen_crop(pil_image: Image.Image, kalmar_id: str, region_idx: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(today_folder("container_results/qwen_images"), f"{kalmar_id}_{timestamp}_r{region_idx}.jpg")
    pil_image.save(path, "JPEG", quality=95)
    print(f"[SAVE] 🧠 QWEN input -> {path}")
    return path

def save_success_result(kalmar_id: str, raw_text: str, frame: np.ndarray):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = today_folder("container_results/success")
    image_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    json_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"kalmar_id": kalmar_id, "raw_ocr_text": raw_text, "timestamp": timestamp}, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] ✅ SUCCESS -> {image_path} + {json_path}")
    return image_path, json_path

# -----------------------------
# Load YOLO models (CPU)
# -----------------------------
def load_yolo(path: str, tag: str = "YOLO"):
    try:
        model = YOLO(path)
        try:
            model.to("cpu")
        except Exception:
            pass
        try:
            model.fuse()
        except Exception:
            # fuse might not be necessary / available depending on ultralytics version
            pass
        print(f"[{tag}] ✅ Loaded: {path}")
        return model
    except Exception as e:
        print(f"[{tag}] ❌ Failed to load ({path}): {e}")
        return None

yolo_primary = load_yolo(YOLO_PRIMARY, "YOLO-PRIMARY")
yolo_secondary = load_yolo(YOLO_SECONDARY, "YOLO-SECONDARY")

# -----------------------------
# Load OpenVINO QWEN via Optimum-Intel (CPU)
# Processor/tokenizer -> from HF (or local HF folder)
# Model (weights) -> from OpenVINO IR folder (INT8 weight-only)
# -----------------------------
print("[QWEN-OV] 🚀 Loading OpenVINO Qwen3-VL (INT8 weight-only)...")

# 1) load processor (tokenizer + image processor) from HF
processor = None
try:
    print(f"[QWEN-OV] Loading processor/tokenizer from HF path: {QWEN_HF_PATH}")
    processor = AutoProcessor.from_pretrained(QWEN_HF_PATH, trust_remote_code=True)
    print("[QWEN-OV] ✅ Processor loaded from HF")
except Exception as e_proc:
    # fallback: try to load processor from the OV folder (if you exported it there)
    try:
        print(f"[QWEN-OV] Processor load from HF failed ({e_proc}). Trying local OV folder for processor: {QWEN_OV_PATH}")
        processor = AutoProcessor.from_pretrained(QWEN_OV_PATH, trust_remote_code=True)
        print("[QWEN-OV] ✅ Processor loaded from OV folder (local)")
    except Exception as e2:
        msg = f"Failed to load processor/tokenizer from both HF ({QWEN_HF_PATH}) and OV folder ({QWEN_OV_PATH}).\nHF error: {e_proc}\nOV-folder error: {e2}\n"
        print("[QWEN-OV] ❌ " + msg)
        raise RuntimeError(msg)

# 2) load OpenVINO model IR (weights)
try:
    qwen_model = OVModelForVisualCausalLM.from_pretrained(QWEN_OV_PATH, device="CPU", trust_remote_code=True)
    print("[QWEN-OV] ✅ OpenVINO model loaded (INT8 weight-only)")
except Exception as e:
    msg = f"Failed to load OpenVINO model from {QWEN_OV_PATH}: {e}"
    print("[QWEN-OV] ❌ " + msg)
    raise RuntimeError(msg)

# concurrency control for model calls
qwen_semaphore = threading.Semaphore(QWEN_MAX_CONCURRENCY)

# -----------------------------
# Assistant response extraction (robust)
# -----------------------------
def extract_assistant_response(full_text: str) -> str:
    if full_text is None:
        return ""
    # pattern 1: <|im_start|>assistant ... <|im_end|>
    if "<|im_start|>assistant" in full_text:
        parts = full_text.split("<|im_start|>assistant")
        if len(parts) > 1:
            response = parts[-1].split("<|im_end|>")[0].strip()
            return response
    # pattern 2: assistant\n
    if "assistant\n" in full_text.lower():
        parts = full_text.lower().split("assistant\n")
        if len(parts) > 1:
            response = full_text.split("assistant\n")[-1].strip()
            return response
    # fallback removal of system blocks
    patterns_to_remove = [r"<\|im_start\|>.*?<\|im_end\|>", r"system:.*?assistant:.*?"]
    cleaned = full_text
    for p in patterns_to_remove:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()

# -----------------------------
# QWEN OCR using Optimum + OpenVINO (CPU) — returns RAW assistant text
# -----------------------------
def qwen_ocr_raw(pil_image: Image.Image) -> str:
    prompt = "Read the container number text and return ONLY the exact characters or text you see."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
        ]
    }]

    # Build template & inputs
    try:
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # some processor versions differ; fallback to simple prompt
        text_prompt = prompt

    inputs = processor(text=[text_prompt], images=[pil_image], return_tensors="pt")

    # Generate (thread-safe)
    with qwen_semaphore:
        start_time = time.time()
        output = qwen_model.generate(**inputs, max_new_tokens=50)
        elapsed = round(time.time() - start_time, 3)

    # Decode robustly
    try:
        out_ids = output[0]
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        if hasattr(out_ids, "tolist"):
            if out_ids.ndim == 2:
                generated_ids = out_ids[0][prompt_len:].tolist()
            else:
                generated_ids = out_ids[prompt_len:].tolist()
            # Prefer processor.decode if available
            if hasattr(processor, "decode"):
                decoded = processor.decode(generated_ids, skip_special_tokens=True)
            elif hasattr(processor, "tokenizer"):
                decoded = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                decoded = str(generated_ids)
        else:
            decoded = str(out_ids)
    except Exception:
        try:
            decoded = processor.decode(output[0], skip_special_tokens=True)
        except Exception:
            decoded = str(output)

    extracted = extract_assistant_response(decoded)

    print(f"[QWEN-OV] ⏱️ Time: {elapsed}s")
    print(f"[QWEN-OV] RAW-DECODE: {repr(decoded)}")
    print(f"[QWEN-OV] EXTRACTED (RAW): {repr(extracted)}")

    return extracted

# -----------------------------
# YOLO detection (single best box)
# -----------------------------
def detect_with_yolo(model, frame: np.ndarray):
    annotated = frame.copy()
    regions = []
    if model is None:
        return False, [], annotated

    results = model(frame, conf=0.15, verbose=False)[0]

    if not results.boxes or len(results.boxes) == 0:
        return False, [], annotated

    # choose highest-confidence box
    best_box = max(results.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    conf = float(best_box.conf)

    # clamp coords
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return False, [], annotated

    crop = frame[y1:y2, x1:x2]

    if crop.size > 0:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        regions.append(Image.fromarray(crop_rgb))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return bool(regions), regions, annotated

# -----------------------------
# Main processing pipeline
# -----------------------------
def process_container_image(base64_image: str, kalmar_id: str) -> dict:
    start_time = time.time()

    try:
        frame = base64_to_cv2(base64_image)
        save_received_frame(frame, kalmar_id)
    except Exception as e:
        print(f"[ERROR] Image decode failed: {e}")
        return {"success": False, "error": "Invalid image data", "processing_time": round(time.time() - start_time, 3)}

    detected, regions, annotated = detect_with_yolo(yolo_primary, frame)

    if not detected and yolo_secondary is not None:
        print("[PIPELINE] 🔄 Trying secondary YOLO...")
        detected, regions, annotated = detect_with_yolo(yolo_secondary, frame)

    save_yolo_detection(annotated, kalmar_id, detected)

    if not detected:
        print("[PIPELINE] ❌ No YOLO detection")
        return {"success": False, "processing_time": round(time.time() - start_time, 3)}

    # Call QWEN for each detected region (stop at first non-empty raw_text)
    for idx, region_pil in enumerate(regions, 1):
        save_qwen_crop(region_pil, kalmar_id, idx)

        try:
            raw_text = qwen_ocr_raw(region_pil)
        except Exception as e:
            print(f"[ERROR] QWEN inference failed: {e}")
            raw_text = ""

        print(f"[OCR] Region {idx} => RAW: {repr(raw_text)}")

        if raw_text and len(raw_text.strip()) > 0:
            save_success_result(kalmar_id, raw_text, frame)
            return {
                "success": True,
                "raw_text": raw_text,
                "processing_time": round(time.time() - start_time, 3)
            }

    return {"success": False, "processing_time": round(time.time() - start_time, 3)}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Container OCR RAW API (OpenVINO INT8 weight-only)", version="1.0")

@app.get("/")
def root():
    return {
        "name": "Container OCR RAW API",
        "version": "1.0",
        "philosophy": "YOLO -> Qwen (OpenVINO INT8 weight-only) — minimal processing",
        "endpoints": {
            "/api/health": "GET - Health check",
            "/api/pickup/event": "POST - Process container images",
            "/api/test-decode": "POST - Test base64 decoding"
        }
    }

@app.get("/api/health")
def health_check():
    core_devices = []
    try:
        import openvino as ov
        core = ov.Core()
        core_devices = list(core.available_devices)
    except Exception:
        core_devices = []

    return {
        "status": "running",
        "runtime": "openvino+optimum-intel",
        "device": "CPU",
        "openvino_devices": core_devices,
        "models": {
            "qwen_openvino": True,
            "yolo_primary": yolo_primary is not None,
            "yolo_secondary": yolo_secondary is not None
        }
    }

@app.post("/api/test-decode")
async def test_decode(data: dict):
    try:
        b64 = data.get("image_base64", "")
        frame = base64_to_cv2(b64)
        return {"status": "success", "message": "Image decoded successfully", "shape": frame.shape, "dtype": str(frame.dtype)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/pickup/event")
async def pickup_event(event: PickupEvent):
    print(f"\n{'='*60}")
    print(f"[API] 📥 Kalmar ID: {event.kalmar_id}")
    print(f"[API] 🕐 Time: {event.timestamp}")
    print(f"{'='*60}")

    images = event.images or ([event.image_base64] if event.image_base64 else [])

    if not images:
        raise HTTPException(status_code=400, detail="No images provided. Include 'images' array or 'image_base64' field.")

    print(f"[API] 📷 Processing {len(images)} image(s)")

    for idx, image_b64 in enumerate(images, 1):
        print(f"\n[API] 🔍 Image {idx}/{len(images)}")
        result = process_container_image(image_b64, event.kalmar_id)

        if result.get("success"):
            print(f"[API] ✅ Found RAW: {result['raw_text']}")
            print(f"[API] ⏱️ Time: {result['processing_time']}s")
            return {
                "status": "container_found",
                "raw_text": result["raw_text"],
                "kalmar_id": event.kalmar_id,
                "processing_time": result["processing_time"]
            }

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

    print("[API] ❌ No container found")
    return {"status": "no_container", "kalmar_id": event.kalmar_id}

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Container OCR RAW API — starting (OpenVINO INT8 weight-only)")
    print("="*60 + "\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
