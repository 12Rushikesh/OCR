# ==========================================================
# containerNoRead_FLORENCE_OCR_CPU_fixed.py
# CPU/GPU adaptive pipeline: YOLO -> Florence-2-OCR -> ISO 6346 validation
# Improvements:
# - extractor returns (container, iso_valid)
# - pipeline accepts ISO-invalid OCR results but marks them
# - prefers ISO-valid candidate if present
# ==========================================================

import os
import re
import time
import json
import base64
import threading
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple

import cv2
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# ==========================================================
# DEVICE CONFIGURATION
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 FORCE CPU ONLY
#DEVICE = "cpu"


# CRITICAL: Florence-2 requires float32 for stable results
# float16 can cause NaN outputs even on GPU
DTYPE = torch.float32

# Set safe CPU thread limits (Windows-friendly)
if DEVICE == "cpu":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

# ==========================================================
# CONFIG
# ==========================================================
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"
FLORENCE_MODEL_DIR = r"E:\ocr\flocr\models\florence2_large"
YOLO_PRIMARY = r"D:\Rushikesh\project\ContainerModel_22_01_26_ubuntu1.pt"
YOLO_SECONDARY = r"D:\Rushikesh\project\ContainerModel_12_01_26_3.pt"

# Performance settings adapted for Florence-2
OCR_MAX_TOKENS = 1024
MAX_IMAGE_SIZE = 1280 if DEVICE == "cpu" else 2048
CONCURRENT_OCR_CALLS = 1 if DEVICE == "cpu" else 2
YOLO_BATCH_SIZE = 1

# ==========================================================
# ENSURE FLORENCE-2 MODEL IS DOWNLOADED
# ==========================================================
def ensure_florence():
    if not os.path.exists(FLORENCE_MODEL_DIR) or not os.listdir(FLORENCE_MODEL_DIR):
        print(f"[FLORENCE] ⬇️ Downloading Florence-2-base from {FLORENCE_MODEL_ID}...")
        snapshot_download(
            repo_id=FLORENCE_MODEL_ID,
            local_dir=FLORENCE_MODEL_DIR,
            local_dir_use_symlinks=False,  # Important for Windows
            resume_download=True
        )
        print(f"[FLORENCE] ✅ Download complete")

print(f"\n{'='*60}")
print(f"[SYSTEM] 🚀 Initializing Florence-2 OCR Pipeline")
print(f"{'='*60}\n")

ensure_florence()

florence_processor = None
florence_model = None

# Load Florence-2 model (processor first)
print("[FLORENCE] 📦 Loading processor...")
try:
    # Force slow tokenizer backend: Florence's processor expects the slow tokenizer API
    florence_processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_DIR,
        trust_remote_code=True,
        use_fast=False  # <-- critical fix to avoid TokenizersBackend mismatch
    )
    print("[FLORENCE] ✅ Processor loaded with use_fast=False (slow tokenizer)")
except Exception as e:
    # Fallback: try default loader but warn the user
    print(f"[FLORENCE] ⚠️ Failed to load processor with use_fast=False: {e}")
    print("[FLORENCE] ⬇️ Falling back to default processor (use_fast=True)")
    florence_processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL_DIR,
        trust_remote_code=True,
        use_fast=True
    )
    print("[FLORENCE] ⚠️ Processor loaded with use_fast=True (fallback) - watch warnings")

# Optional diagnostic: show tokenizer class
try:
    tok_cls = florence_processor.tokenizer.__class__.__name__
    print(f"[FLORENCE] Tokenizer class: {tok_cls}")
except Exception:
    pass

print(f"[FLORENCE] 🧠 Loading model to {DEVICE.upper()}...")
florence_model = AutoModelForCausalLM.from_pretrained(
    FLORENCE_MODEL_DIR,
    torch_dtype=DTYPE,
    trust_remote_code=True
).to(DEVICE)

florence_model.eval()
print(f"[FLORENCE] ✅ Florence-2 loaded on {DEVICE.upper()}")

# ==========================================================
# SYSTEM INFO
# ==========================================================
print(f"\n{'='*60}")
print(f"[SYSTEM] Configuration Summary")
print(f"{'='*60}")
print(f"Device: {DEVICE}")
print(f"Dtype: {DTYPE}")
print(f"Model: Florence-2-base")
print(f"Max tokens: {OCR_MAX_TOKENS}")
print(f"Concurrent calls: {CONCURRENT_OCR_CALLS}")
print(f"Max image size: {MAX_IMAGE_SIZE}px")
print(f"YOLO batch size: {YOLO_BATCH_SIZE}")
print(f"{'='*60}\n")

# Thread-safe semaphore for concurrent OCR calls
florence_semaphore = threading.Semaphore(CONCURRENT_OCR_CALLS)

# ==========================================================
# DIRECTORY SETUP
# ==========================================================
def today_folder(base: str) -> str:
    folder_path = os.path.join(base, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

for directory in [
    "container_results/received_frames",
    "container_results/yolo_detections",
    "container_results/florence_images",
    "container_results/success"
]:
    os.makedirs(directory, exist_ok=True)

# ==========================================================
# PYDANTIC MODELS
# ==========================================================
class PickupEvent(BaseModel):
    kalmar_id: str
    action: str
    timestamp: str
    images: Optional[List[str]] = None
    image_base64: Optional[str] = None

# ==========================================================
# LOAD YOLO MODELS
# ==========================================================
def load_yolo(path: str, tag: str = "YOLO"):
    try:
        model = YOLO(path)
        try:
            model.fuse()
        except Exception:
            pass

        if DEVICE == "cpu":
            try:
                model.to("cpu")
            except Exception:
                pass

        print(f"[{tag}] ✅ Loaded on {DEVICE.upper()}: {path}")
        return model
    except Exception as e:
        print(f"[{tag}] ❌ Failed to load ({path}): {e}")
        return None

yolo_primary = load_yolo(YOLO_PRIMARY, "YOLO-PRIMARY")
yolo_secondary = load_yolo(YOLO_SECONDARY, "YOLO-SECONDARY")

# ==========================================================
# UTILS: base64 -> cv2
# ==========================================================
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

# ==========================================================
# SAVE HELPERS
# ==========================================================
def save_received_frame(frame: np.ndarray, kalmar_id: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(
        today_folder("container_results/received_frames"),
        f"{kalmar_id}_{timestamp}.jpg"
    )
    cv2.imwrite(path, frame)
    print(f"[SAVE] 📷 Received -> {path}")
    return path

def save_yolo_detection(frame: np.ndarray, kalmar_id: str, detected: bool):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    status = "detected" if detected else "no_detection"
    path = os.path.join(
        today_folder("container_results/yolo_detections"),
        f"{kalmar_id}_{timestamp}_{status}.jpg"
    )
    cv2.imwrite(path, frame)
    print(f"[SAVE] 🎯 YOLO -> {path}")
    return path

def save_florence_crop(pil_image: Image.Image, kalmar_id: str, region_idx: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(
        today_folder("container_results/florence_images"),
        f"{kalmar_id}_{timestamp}_r{region_idx}.jpg"
    )
    pil_image.save(path, "JPEG", quality=95)
    print(f"[SAVE] 🧠 Florence input -> {path}")
    return path

def save_success_result(kalmar_id: str, container_number: str, frame: np.ndarray):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = today_folder("container_results/success")

    image_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)

    json_path = os.path.join(base_folder, f"{kalmar_id}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            "kalmar_id": kalmar_id,
            "container_number": container_number,
            "iso6346_valid": iso6346_checksum(container_number),
            "timestamp": timestamp
        }, f, indent=2)

    print(f"[SAVE] ✅ SUCCESS -> {image_path} + {json_path}")
    return image_path, json_path

# ==========================================================
# ISO 6346 CHECKSUM VALIDATION
# ==========================================================
ISO_CHAR_MAP = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19,
    'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29,
    'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
}

def iso6346_checksum(container: str) -> bool:
    if not container or len(container) != 11:
        return False

    if not (container[:4].isalpha() and container[4:].isdigit()):
        return False

    total = 0
    try:
        for i in range(10):
            ch = container[i]
            val = int(ch) if ch.isdigit() else ISO_CHAR_MAP.get(ch, 0)
            total += val * (2 ** i)

        remainder = total % 11
        check_digit = 0 if remainder == 10 else remainder

        return check_digit == int(container[10])
    except Exception:
        return False

def extract_container_from_text(text: str) -> Tuple[str, bool]:
    """
    Returns (container_candidate, iso_valid).
    - Prefers and returns the first ISO-valid candidate with iso_valid=True.
    - If none are ISO-valid but there are candidates, returns the first candidate with iso_valid=False.
    - If no candidate found, returns ("", False).
    """
    if not text:
        return "", False

    # Remove everything except letters & digits
    cleaned = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    candidates = re.findall(r"[A-Z]{4}\d{7}", cleaned)

    if not candidates:
        print(f"[VALIDATION] ❌ No container pattern in: {repr(text)}")
        return "", False

    # Prefer a valid candidate
    for candidate in candidates:
        if iso6346_checksum(candidate):
            print(f"[VALIDATION] ✅ Valid ISO6346: {candidate}")
            return candidate, True

    # No valid candidate — return first candidate but mark invalid
    first = candidates[0]
    print(f"[VALIDATION] ⚠️ No ISO-valid candidate, accepting OCR: {first}")
    return first, False

# ==========================================================
# FLORENCE-2 OCR
# ==========================================================
def florence_ocr_raw(pil_image: Image.Image) -> Tuple[str, bool]:
    """
    Florence-2 OCR. Returns tuple (container_candidate, iso_valid).
    If no candidate found, returns ("", False).
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize if needed
    if max(pil_image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        print(f"[FLORENCE] ⚡ Resized to {new_size}")

    width, height = pil_image.size

    # Florence-2 task prompt - use <OCR> for text extraction
    task_prompt = "<OCR>"

    # Prepare inputs
    inputs = florence_processor(
        text=task_prompt,
        images=pil_image,
        return_tensors="pt"
    )

    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    start_time = time.time()
    try:
        with torch.no_grad():
            generated_ids = florence_model.generate(
                input_ids=inputs.get("input_ids"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=OCR_MAX_TOKENS,
                do_sample=False,
                num_beams=3  # Beam search for better accuracy
            )

        # Decode output
        generated_text = florence_processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        # CRITICAL: Florence-2 post-processing
        parsed_answer = florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(width, height)
        )

        # Extract OCR text from parsed result
        ocr_text = parsed_answer.get(task_prompt, "")

        elapsed = round(time.time() - start_time, 3)

        print(f"[FLORENCE] ⏱️ Time: {elapsed}s ({DEVICE.upper()})")
        print(f"[FLORENCE] RAW-OUTPUT: {repr(generated_text)}")
        print(f"[FLORENCE] OCR-TEXT: {repr(ocr_text)}")

        # Validate and extract container number (may be ISO-invalid)
        container_candidate, iso_valid = extract_container_from_text(ocr_text)

        if container_candidate:
            if iso_valid:
                print(f"[FLORENCE] ✅ VALIDATED: {container_candidate}")
            else:
                print(f"[FLORENCE] ⚠️ OCR-only (checksum failed): {container_candidate}")
            return container_candidate, iso_valid
        else:
            print(f"[FLORENCE] ❌ No valid container found")
            return "", False

    except Exception as e:
        print(f"[FLORENCE] ❌ Generation Error: {e}")
        import traceback
        traceback.print_exc()
        return "", False

def safe_florence_ocr_raw(pil_image: Image.Image) -> Tuple[str, bool]:
    with florence_semaphore:
        return florence_ocr_raw(pil_image)

# ==========================================================
# YOLO DETECTION
# ==========================================================
def detect_with_yolo(model, frame: np.ndarray):
    """YOLO detection."""
    annotated = frame.copy()
    regions = []

    if model is None:
        return False, [], annotated

    try:
        results = model(frame, conf=0.15, verbose=False)[0]
    except Exception as e:
        print(f"[YOLO] ❌ Inference error: {e}")
        return False, [], annotated

    if not results.boxes or len(results.boxes) == 0:
        return False, [], annotated

    best_box = max(results.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    conf = float(best_box.conf)

    crop = frame[y1:y2, x1:x2]

    if crop.size > 0:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        regions.append(Image.fromarray(crop_rgb))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return bool(regions), regions, annotated

# ==========================================================
# MAIN PROCESSING PIPELINE
# ==========================================================
def process_container_image(base64_image: str, kalmar_id: str) -> dict:
    start_time = time.time()

    try:
        frame = base64_to_cv2(base64_image)
        save_received_frame(frame, kalmar_id)
    except Exception as e:
        print(f"[ERROR] Image decode failed: {e}")
        return {
            "success": False,
            "error": "Invalid image data",
            "processing_time": round(time.time() - start_time, 3)
        }

    detected, regions, annotated = detect_with_yolo(yolo_primary, frame)

    if not detected and yolo_secondary is not None:
        print("[PIPELINE] 🔄 Trying secondary YOLO...")
        detected, regions, annotated = detect_with_yolo(yolo_secondary, frame)

    save_yolo_detection(annotated, kalmar_id, detected)

    if not detected:
        print("[PIPELINE] ❌ No YOLO detection")
        return {
            "success": False,
            "processing_time": round(time.time() - start_time, 3)
        }

    for idx, region_pil in enumerate(regions, 1):
        save_florence_crop(region_pil, kalmar_id, idx)

        container_candidate, iso_valid = safe_florence_ocr_raw(region_pil)
        print(f"[OCR] Region {idx} => CANDIDATE: {repr(container_candidate)}, iso_valid={iso_valid}")

        if container_candidate and len(container_candidate) == 11:
            save_success_result(kalmar_id, container_candidate, frame)

            return {
                "success": True,
                "container_number": container_candidate,
                "iso6346_valid": iso_valid,
                "device": DEVICE,
                "processing_time": round(time.time() - start_time, 3)
            }

    return {
        "success": False,
        "processing_time": round(time.time() - start_time, 3)
    }

# ==========================================================
# FASTAPI APP
# ==========================================================
app = FastAPI(
    title="Container OCR Florence-2 API",
    description="YOLO + Florence-2 OCR with ISO 6346 validation",
    version="2.0-FLORENCE"
)

@app.get("/")
def root():
    return {
        "name": "Container OCR Florence-2 API",
        "version": "2.0-FLORENCE",
        "model": "Florence-2-base",
        "device": DEVICE,
        "dtype": str(DTYPE),
        "cuda_available": torch.cuda.is_available(),
        "features": [
            "Florence-2 OCR",
            "ISO 6346 checksum validation (flagged if invalid)",
            "Regex extraction [A-Z]{4}\\d{7}",
            "CPU/GPU adaptive inference"
        ],
        "optimizations": {
            "max_image_size": MAX_IMAGE_SIZE,
            "max_tokens": OCR_MAX_TOKENS,
            "concurrent_calls": CONCURRENT_OCR_CALLS,
            "yolo_batch_size": YOLO_BATCH_SIZE,
            "beam_search": True
        },
        "endpoints": {
            "/api/health": "GET - Health check",
            "/api/pickup/event": "POST - Process container images",
            "/api/test-decode": "POST - Test base64 decoding",
            "/api/gpu-status": "GET - GPU status"
        }
    }

@app.get("/api/health")
def health_check():
    gpu_memory = "N/A"
    if torch.cuda.is_available():
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    
    return {
        "status": "running",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": gpu_memory,
        "models": {
            "florence_ocr": florence_model is not None,
            "yolo_primary": yolo_primary is not None,
            "yolo_secondary": yolo_secondary is not None
        }
    }

@app.get("/api/gpu-status")
def gpu_status():
    if not torch.cuda.is_available():
        return {"status": "No GPU available", "device": "cpu"}
    
    return {
        "status": "GPU available",
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
        "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f} GB"
    }

@app.post("/api/test-decode")
async def test_decode(data: dict):
    try:
        b64 = data.get("image_base64", "")
        frame = base64_to_cv2(b64)
        return {
            "status": "success",
            "message": "Image decoded successfully",
            "shape": frame.shape,
            "dtype": str(frame.dtype),
            "device": DEVICE
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/pickup/event")
async def pickup_event(event: PickupEvent):
    print(f"\n{'='*60}")
    print(f"[API] 📥 Kalmar ID: {event.kalmar_id}")
    print(f"[API] 🕐 Time: {event.timestamp}")
    print(f"[API] 🖥️ Device: {DEVICE.upper()}")
    print(f"{'='*60}")

    images = event.images or ([event.image_base64] if event.image_base64 else [])

    if not images:
        raise HTTPException(
            status_code=400,
            detail="No images provided. Include 'images' array or 'image_base64' field."
        )

    print(f"[API] 📷 Processing {len(images)} image(s)")

    for idx, image_b64 in enumerate(images, 1):
        print(f"\n[API] 🔍 Image {idx}/{len(images)}")
        result = process_container_image(image_b64, event.kalmar_id)

        if result.get("success"):
            print(f"[API] ✅ Container: {result['container_number']}")
            print(f"[API] ⏱️ Time: {result['processing_time']}s on {result.get('device', DEVICE).upper()}")

            return {
                "status": "container_found",
                "container_number": result["container_number"],
                "iso6346_valid": result["iso6346_valid"],
                "kalmar_id": event.kalmar_id,
                "processing_time": result["processing_time"],
                "device": result.get("device", DEVICE)
            }

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

    print("[API] ❌ No container found")
    return {
        "status": "no_container",
        "kalmar_id": event.kalmar_id,
        "device": DEVICE
    }

# ==========================================================
# RUN SERVER
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 Container OCR Florence-2 API")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
