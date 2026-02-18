# ==========================================================
# containerNoRead.py — SINGLE YOLO + QWEN (CPU OPTIMIZED)
# ==========================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # ✅ Force CPU — no accidental CUDA usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"                 # ✅ Reduced from 2 → 1 (less contention)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"          # ✅ Added (macOS/some Windows builds)

import cv2
import base64
import json
import re
import gc                                            # ✅ Added — explicit memory cleanup
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import threading
import time
from datetime import datetime
from transformers import AutoProcessor
from optimum.intel import OVModelForVisualCausalLM

# ==========================================================
# BASE DIRECTORY (SERVICE SAFE)
# ==========================================================
BASE_DIR = "container_results"

RECEIVED_DIR = os.path.join(BASE_DIR, "received_frames")
YOLO_DIR     = os.path.join(BASE_DIR, "yolo_detections")
QWEN_DIR     = os.path.join(BASE_DIR, "qwen_images")
SUCCESS_DIR  = os.path.join(BASE_DIR, "success")

for d in [RECEIVED_DIR, YOLO_DIR, QWEN_DIR, SUCCESS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# CONFIG
# ==========================================================
QWEN_OV_PATH    = r"C:\YARDOPERATION\qwen3-vl-2b-int8-weightonly-ov"
QWEN_HF_PATH    = "Qwen/Qwen3-VL-2B-Instruct"
YOLO_MODEL_PATH = r"C:\YARDOPERATION\ContainerModel_12_01_26_3.pt"

ISO_BASE_REGEX = re.compile(r"[A-Z]{3}U\d{6}")
ISO_11_REGEX   = re.compile(r"[A-Z]{3}U\d{7}")

# ==========================================================
# ISO 6346 CHECK DIGIT
# ==========================================================
LETTER_VALUES = [
    10,12,13,14,15,16,17,18,19,20,21,
    23,24,25,26,27,28,29,30,31,32,
    34,35,36,37,38
]

def compute_check_digit(code10: str) -> int:
    total = 0
    for i, c in enumerate(code10):
        if i < 4:
            value = LETTER_VALUES[ord(c) - 65]
        else:
            value = int(c)
        total += value * (1 << i)
    remainder = total % 11
    return 0 if remainder == 10 else remainder

def extract_and_validate_container(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    # Fix OCR confusion in owner code
    chars = list(text)
    for i in range(min(4, len(chars))):
        if chars[i] == "0":
            chars[i] = "O"
    text = "".join(chars)

    # If full 11-digit found → return
    m11 = ISO_11_REGEX.search(text)
    if m11:
        return m11.group()

    # If 10-digit found → compute digit
    m10 = ISO_BASE_REGEX.search(text)
    if m10:
        base = m10.group()
        digit = compute_check_digit(base)
        return base + str(digit)

    return ""

# ==========================================================
# LOAD QWEN
# ==========================================================
print("[QWEN-OV] 🚀 Loading OpenVINO INT8 Qwen3-VL...")

# Load processor (tokenizer + image processor)
try:
    qwen_processor = AutoProcessor.from_pretrained(QWEN_HF_PATH, trust_remote_code=True)
    print("[QWEN-OV] ✅ Processor loaded from HF")
except Exception:
    qwen_processor = AutoProcessor.from_pretrained(QWEN_OV_PATH, trust_remote_code=True)
    print("[QWEN-OV] ✅ Processor loaded from OV folder")

# Load OpenVINO model (INT8 IR)
qwen_model = OVModelForVisualCausalLM.from_pretrained(
    QWEN_OV_PATH,
    device="CPU",
    trust_remote_code=True
)

print("[QWEN-OV] ✅ Model loaded (CPU INT8)")

qwen_semaphore = threading.Semaphore(1)

# ==========================================================
# LOAD YOLO (single model — CPU)
# ==========================================================
print("[YOLO] 🚀 Loading...")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    try:
        yolo_model.to("cpu")        # ✅ Explicitly pin to CPU
    except Exception:
        pass
    try:
        yolo_model.fuse()           # ✅ Fuse layers → faster CPU inference, lower memory
    except Exception:
        pass
    print("[YOLO] ✅ Loaded")
except Exception as e:
    print("[YOLO] ❌ Failed:", e)
    yolo_model = None

# ==========================================================
# REQUEST MODEL
# ==========================================================
class PickupEvent(BaseModel):
    kalmar_id: str
    action: str
    timestamp: str
    images: Optional[List[str]] = None
    image_base64: Optional[str] = None

# ==========================================================
# UTILITIES
# ==========================================================
def today_folder(base):
    path = os.path.join(base, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(path, exist_ok=True)
    return path

def base64_to_cv2(b64: str):
    try:
        if b64.startswith("data:"):             # ✅ Handle data-URI prefix
            b64 = b64.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None — invalid image data")
        return frame
    except Exception as e:
        print(f"[ERROR] base64_to_cv2 failed: {e}")
        raise

# ==========================================================
# SAVE HELPERS (ABSOLUTE PATH SAFE)
# ==========================================================
def save_received(frame, kid):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    p = os.path.join(today_folder(RECEIVED_DIR), f"{kid}_{ts}.jpg")
    cv2.imwrite(p, frame)

def save_yolo(frame, kid, detected):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    s = "detected" if detected else "no_detection"
    p = os.path.join(today_folder(YOLO_DIR), f"{kid}_{ts}_{s}.jpg")
    cv2.imwrite(p, frame)

def save_qwen_img(pil, kid):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    p = os.path.join(today_folder(QWEN_DIR), f"{kid}_{ts}.jpg")
    pil.save(p, "JPEG", quality=85)             # ✅ quality 95 → 85 (saves disk I/O, no OCR impact)

def save_success(kid, iso, frame, raw):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = today_folder(SUCCESS_DIR)

    img = os.path.join(base, f"{kid}_{iso}_{ts}.jpg")
    js  = os.path.join(base, f"{kid}_{iso}_{ts}.json")

    cv2.imwrite(img, frame)
    with open(js, "w") as f:
        json.dump({
            "kalmar_id": kid,
            "container_number": iso,
            "raw_text": raw,
            "timestamp": ts
        }, f, indent=2)

# ==========================================================
# QWEN OCR
# ==========================================================
def qwen_ocr(pil: Image.Image):

    prompt = "Read the container number and return ONLY the exact characters."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": prompt}
        ]
    }]

    try:
        text_prompt = qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        text_prompt = prompt

    inputs = qwen_processor(
        text=[text_prompt],
        images=[pil],
        return_tensors="pt"
    )

    with qwen_semaphore:
        start = time.time()
        output = qwen_model.generate(**inputs, max_new_tokens=20)
        print("[QWEN-OV] ⏱️ Time:", round(time.time() - start, 3), "sec")

    # ✅ Free inputs right after generate — reclaims memory faster
    del inputs
    gc.collect()

    # Decode safely
    try:
        out_ids = output[0]
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_ids = out_ids[prompt_len:].tolist()
        full_text = qwen_processor.decode(generated_ids, skip_special_tokens=True)
    except Exception:
        full_text = qwen_processor.decode(output[0], skip_special_tokens=True)

    iso = extract_and_validate_container(full_text)

    print("[QWEN RAW]:", full_text)
    print("[QWEN FINAL ISO]:", iso)

    return iso, full_text


def safe_qwen_ocr(pil):
    # NOTE: semaphore is already acquired inside qwen_ocr — no double-wrap needed
    return qwen_ocr(pil)

# ==========================================================
# YOLO + OCR PIPELINE
# ==========================================================
def detect_yolo(frame):
    if yolo_model is None:
        return False, None, frame           # ✅ No copy when no model — saves RAM

    res = yolo_model(frame, conf=0.25, verbose=False)[0]

    if not res.boxes or len(res.boxes) == 0:
        return False, None, frame           # ✅ No copy when no detection — saves RAM

    best = max(res.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = map(int, best.xyxy[0])

    # ✅ Clamp coords to frame bounds (prevents crashes on edge detections)
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return False, None, frame

    crop = frame[y1:y2, x1:x2]

    ann = frame.copy()                      # ✅ Only copy when we actually have a box to draw
    cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if crop.size == 0:
        return False, None, ann

    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return True, pil, ann

def process_container_image(b64, kid):
    start = time.time()

    frame = base64_to_cv2(b64)
    save_received(frame, kid)

    detected, region, ann = detect_yolo(frame)
    save_yolo(ann, kid, detected)

    if not detected:
        # ✅ Free memory before early return
        del frame, ann
        gc.collect()
        return {"success": False}

    save_qwen_img(region, kid)
    iso, raw = safe_qwen_ocr(region)

    if iso:
        save_success(kid, iso, frame, raw)
        result = {
            "success": True,
            "iso_code": iso,
            "processing_time": round(time.time() - start, 3)
        }
        del frame, ann, region
        gc.collect()
        return result

    del frame, ann, region
    gc.collect()
    return {"success": False}

# ==========================================================
# FASTAPI
# ==========================================================
app = FastAPI(title="Container OCR API (Qwen + Single YOLO)")

@app.post("/api/pickup/event")
async def pickup_event(event: PickupEvent):

    imgs = event.images or ([event.image_base64] if event.image_base64 else [])
    if not imgs:
        raise HTTPException(status_code=400, detail="No images provided")

    for img in imgs:
        res = process_container_image(img, event.kalmar_id)
        if res.get("success"):
            return {
                "status": "container_found",
                "container_number": res["iso_code"],
                "kalmar_id": event.kalmar_id
            }

    return {
        "status": "no_container",
        "kalmar_id": event.kalmar_id
    }

@app.get("/api/health")
def health():
    core_devices = []
    try:
        import openvino as ov
        core = ov.Core()
        core_devices = list(core.available_devices)
    except Exception:
        pass

    return {
        "status": "running",
        "device": "CPU",
        "openvino_devices": core_devices,
        "qwen_loaded": True,
        "yolo_loaded": yolo_model is not None
    }

# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
